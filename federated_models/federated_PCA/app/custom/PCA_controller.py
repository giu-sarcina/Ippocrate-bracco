import gc

from nvflare.apis.shareable import Shareable
from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.dxo import DataKind
from nvflare.apis.signal import Signal
from nvflare.apis.fl_context import FLContext
from nvflare.apis.controller_spec import OperatorMethod, Task, TaskOperatorKey

from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.security.logging import secure_format_exception

class PassthroughShareableGenerator(ShareableGenerator):
    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> Learnable:
        result = Learnable()
        for k, v in shareable.items():
            result[k] = v
        return result

    def learnable_to_shareable(self, model: Learnable, fl_ctx: FLContext) -> Shareable:
        result = Shareable()
        for k, v in model.items():
            result[k] = v
        return result


class PCAController(ScatterAndGather):
    def __init__ (self,results_path:str, 
        min_clients: int = 1,
        num_rounds: int = 1,
        start_round: int = 0,
        wait_time_after_min_received: int = 10,
        aggregator_id = Aggregator,
        persistor_id= NPModelPersistor,
        shareable_generator_id = FullModelShareableGenerator,
        train_task_name="PCA_fit",
        task_check_period: float = 0.5,
        train_timeout: int = 0):
        
        super().__init__(task_check_period=task_check_period)
                
        self.aggregator_id = aggregator_id
        self.persistor_id = persistor_id
        self.shareable_generator_id = shareable_generator_id
        self.train_task_name = train_task_name
        self.aggregator = None # from config
        self.persistor = None
        self.shareable_gen = None
        
        self.results_path = results_path
        
        self._min_clients = min_clients
        self._num_rounds = num_rounds
        self._start_round = start_round
        self._wait_time_after_min_received = wait_time_after_min_received
        
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:
            self.log_info(fl_ctx, "Beginning ScatterAndGather training phase.")
            self._phase = AppConstants.PHASE_TRAIN

            fl_ctx.set_prop(AppConstants.PHASE, self._phase, private=True, sticky=False)
            fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            if self._current_round is None:
                self._current_round = self._start_round
            while self._current_round < self._start_round + self._num_rounds:

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return
                
                # TASK 1: Standardization
                # Create empty shareable to send to clients
                n_mean_std = {}
                n_mean_std["local_count"] = None
                n_mean_std["local_mean"] = None
                n_mean_std["local_std"] = None
                
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=n_mean_std)
                n_mean_std_shareable = dxo.to_shareable()
        
                self.aggregator.task = "Data_standardization"
                self.shareable_gen = FullModelShareableGenerator() #  to convert between shareable and learnable (to use persistor)
                self.persistor = NPModelPersistor(self.results_path, "count_mean_std.npy")
                # Create standardization task
                std_task = Task(
                    name="Data_standardization",
                    data=n_mean_std_shareable,
                    result_received_cb=self._process_train_result,
                    timeout=self._train_timeout,)
                # Send message to all clients
                self.broadcast_and_wait(
                    task=std_task,
                    min_responses=self._min_clients,
                    wait_time_after_min_received=self._wait_time_after_min_received,
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )

                n_mean_std = self.aggregator.global_standardization(fl_ctx)
                
                if self._check_abort_signal(fl_ctx, abort_signal):
                    return
                                
                self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, fl_ctx)
                n_mean_std_learnable = self.shareable_gen.shareable_to_learnable(n_mean_std, fl_ctx)
                fl_ctx.set_prop("count, mean and std", n_mean_std_learnable, private=True, sticky=True)
                fl_ctx.sync_sticky()
                self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, fl_ctx)
                
                if self._check_abort_signal(fl_ctx, abort_signal):
                    return
                
                #Save global mean and std
                if self.persistor:
                    if (
                        self._persist_every_n_rounds != 0
                        and (self._current_round + 1) % self._persist_every_n_rounds == 0
                    ) or self._current_round == self._start_round + self._num_rounds - 1:
                        self.log_info(fl_ctx, "Start persist model on server.")
                        self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
                        self.persistor.save(n_mean_std_learnable, fl_ctx)
                        self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)
                        self.log_info(fl_ctx, "End persist model on server.")
                
                # TASK 2: Local non normalized covariance + global PCA
                self.log_info(fl_ctx, f"Calculating local covariances...")
                self.log_info(fl_ctx, f"Round {self._current_round} started.")
                
                #fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=True)
                self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)
                
                #Create task train 
                cov = {"numpy_key":{ "covariance": None }}
                data = {
                    "n_mean_std": n_mean_std.get("DXO").get("data"),
                    "covariance": cov
                }
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=data)
                data_shareable = dxo.to_shareable()
                
                data_shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
                data_shareable.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
                data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)
                
                operator = {
                    TaskOperatorKey.OP_ID: self.train_task_name,
                    TaskOperatorKey.METHOD: OperatorMethod.BROADCAST,
                    TaskOperatorKey.TIMEOUT: self._train_timeout,
                    TaskOperatorKey.AGGREGATOR: self.aggregator_id,
                }

                train_task = Task(
                    name= "PCA_fit", #self.train_task_name,
                    data=data_shareable,
                    operator=operator,
                    props={},
                    timeout=self._train_timeout,
                    before_task_sent_cb=self._prepare_train_task_data,
                    result_received_cb=self._process_train_result,
                )
                
                # Invia lo stesso messaggio a tutti i client
                self.aggregator.task = "PCA_fit"
                self.persistor = NPModelPersistor(self.results_path, "PCA.npy")
                self.broadcast_and_wait(
                    task=train_task,
                    min_responses=self._min_clients,
                    wait_time_after_min_received=self._wait_time_after_min_received,
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.log_info(fl_ctx, "Start aggregation.")
                self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
                aggr_result = self.aggregator.aggregate(fl_ctx)
                fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
                self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
                self.log_info(fl_ctx, "End aggregation.")

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, fl_ctx)
                aggr_result_learnable = self.shareable_gen.shareable_to_learnable(aggr_result, fl_ctx)
                fl_ctx.set_prop("AGGR_PCA", aggr_result_learnable, private=True, sticky=True)
                
                fl_ctx.sync_sticky()
                self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                if self.persistor:
                    if (
                        self._persist_every_n_rounds != 0
                        and (self._current_round + 1) % self._persist_every_n_rounds == 0
                    ) or self._current_round == self._start_round + self._num_rounds - 1:
                        self.log_info(fl_ctx, "Start persist model on server.")
                        self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
                        self.persistor.save(aggr_result_learnable, fl_ctx)
                        self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)
                        self.log_info(fl_ctx, "End persist model on server.")

                # TASK 3: proiezioni + volumi locali + confronti globali
                self.aggregator.task = "Data_similarity"
                self.persistor = NPModelPersistor(self.results_path, "similarity_matrix.npy")

                transform_task = Task(
                    name="Data_similarity",
                    data=aggr_result,
                    result_received_cb=self._process_train_result,
                    timeout=self._train_timeout,
                )
                self.broadcast_and_wait(
                    task=transform_task,
                    min_responses=self._min_clients,
                    wait_time_after_min_received=self._wait_time_after_min_received,
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )
                
                similarity = self.aggregator.compare(fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return
                
                self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, fl_ctx)
                similarity_learnable = self.shareable_gen.shareable_to_learnable(similarity, fl_ctx)
                fl_ctx.set_prop("SIMILARITY_MODEL", similarity_learnable, private=True, sticky=True)
                fl_ctx.sync_sticky()
                self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, fl_ctx)
                

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return
                
                if self.persistor:
                    if (
                        self._persist_every_n_rounds != 0
                        and (self._current_round + 1) % self._persist_every_n_rounds == 0
                    ) or self._current_round == self._start_round + self._num_rounds - 1:
                        self.log_info(fl_ctx, "Start persist model on server.")
                        self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
                        self.persistor.save(similarity_learnable, fl_ctx)
                        self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)
                        self.log_info(fl_ctx, "End persist model on server.")

                self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                self.log_info(fl_ctx, f"Round {self._current_round} finished.")
                self._current_round += 1

                # need to persist snapshot after round increased because the global weights should be set to
                # the last finished round's result
                if self._snapshot_every_n_rounds != 0 and self._current_round % self._snapshot_every_n_rounds == 0:
                    self._engine.persist_components(fl_ctx, completed=False)

                gc.collect()

            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, "Finished ScatterAndGather Training.")
        except Exception as e:
            error_msg = f"Exception in ScatterAndGather control_flow: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

