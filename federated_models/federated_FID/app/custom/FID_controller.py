import gc
import numpy as np 
from scipy import linalg


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
    

class FIDController(ScatterAndGather):
    def __init__ (self,results_path:str, 
        min_clients: int = 1,
        num_rounds: int = 1,
        start_round: int = 0,
        wait_time_after_min_received: int = 10,
        aggregator_id = Aggregator,
        persistor_id= NPModelPersistor,
        shareable_generator_id = FullModelShareableGenerator,
        train_task_name="FID_to_similarity",
        task_check_period: float = 0.5,
        train_timeout: int = 0):
        
        super().__init__(task_check_period=task_check_period)
                
        self.aggregator_id = aggregator_id
        self.persistor_id = persistor_id
        self.shareable_generator_id = shareable_generator_id
        self.train_task_name = train_task_name
        self.aggregator = None 
        self.persistor = None
        self.shareable_gen = None
        
        self.results_path = results_path
        
        self._min_clients = min_clients
        self._num_rounds = num_rounds
        self._start_round = start_round
        self._wait_time_after_min_received = wait_time_after_min_received
        
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:
            self.log_info(fl_ctx, "Beginning ScatterAndGather.")
            #self._phase = AppConstants.PHASE_TRAIN

            fl_ctx.set_prop(AppConstants.PHASE, self._phase, private=True, sticky=False)
            fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            if self._current_round is None:
                self._current_round = self._start_round
            while self._current_round < self._start_round + self._num_rounds:

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return
                
                self.shareable_gen = FullModelShareableGenerator() 
                
                # TASK 1: Calculate Fréchet distance

                self.log_info(fl_ctx, f"Receiving local inception distribution.. ")
                self.aggregator.task = "FID_to_similarity"
                self.persistor = NPModelPersistor(self.results_path, "FID_to_similarity.npy")
                
                data_shareable = Shareable()
                # Create train task
                train_task = Task(
                    name="FID_to_similarity",
                    data=data_shareable,
                    result_received_cb=self._process_train_result,
                    timeout=self._train_timeout,)
                # Send message to all clients
                self.broadcast_and_wait(
                    task=train_task,
                    min_responses=self._min_clients,
                    wait_time_after_min_received=self._wait_time_after_min_received,
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )
                
                # Calculate distribrution distance 
                aggr_results = self.aggregator.aggregate(fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return
                                
                self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, fl_ctx)
                aggr_results_learnable = self.shareable_gen.shareable_to_learnable(aggr_results, fl_ctx)
                fl_ctx.set_prop(" FID matrix ", aggr_results_learnable, private=True, sticky=True)
                fl_ctx.sync_sticky()
                self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, fl_ctx)
                
                if self._check_abort_signal(fl_ctx, abort_signal):
                    return
                
                #Save FID matrix
                self.log_info(fl_ctx, f"Saving FID matrix to {self.results_path}")
                
                if self.persistor:
                    if (
                        self._persist_every_n_rounds != 0
                        and (self._current_round + 1) % self._persist_every_n_rounds == 0
                    ) or self._current_round == self._start_round + self._num_rounds - 1:
                        self.log_info(fl_ctx, "Start persist model on server.")
                        self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
                        self.persistor.save(aggr_results_learnable, fl_ctx)
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

