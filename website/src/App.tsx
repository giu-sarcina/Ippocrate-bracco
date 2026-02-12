import { useEffect, useState } from "react";
import "./index.css";
import logo from "./assets/logo.svg";
import mouse from "./assets/mouse.svg";
import bracco from "./assets/partners/bracco.svg";
import gaslini from "./assets/partners/gaslini.svg";
import orobix from "./assets/partners/orobix.svg";
import ucbm from "./assets/partners/ucbm.svg";
import gmail from "./assets/gmail.svg";
import outlook from "./assets/outlook.png";
import map from "./assets/map.png";
import iit from "./assets/partners/iit.png";
import mise from "./assets/partners/mise.png";
// import sanmartino from "./assets/partners/sanmartino.png";
import innomed from "./assets/partners/innomed.png";
import appHome from "./assets/app-home.png";
import { EMAIL, NAME } from "./constants";
import { _ } from "node_modules/tailwindcss/dist/colors-b_6i0Oi7";

export function App() {
  const [scrolled, setScrolled] = useState(false);
  const [index, setIndex] = useState(0);

  function nextSlide() {
    setIndex((prev) => (prev + 1 >= slides.length ? 0 : prev + 1));
  }

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const interval = setInterval(nextSlide, 7000);
    return () => clearInterval(interval);
  }, [index]);

  const slides = [
    <div className="flex flex-col lg:flex-row w-screen h-[550px] md:h-[600px] shadow-md">
      <div className="text-[#020e16] h-full py-10 px-6 lg:py-20 lg:px-12 flex-1 overflow-hidden">
        <h1 className="text-[25px] md:text-[40px] lg:text-[50px] font-[800] mb-6 lg:mb-10">
          Data Harmonization & Site Validation
        </h1>
        <span className="text-[16px] lg:text-[18px] font-[350] inline-block">
          {NAME} ensures data integrity in federated learning through automated
          preprocessing, harmonization, and validation for clinical, imaging,
          and omics data. Machine-interpretable protocols standardize workflows
          across sites. Before training, each site undergoes validation using
          federated PCA for structured data and federated FID for imaging to
          detect distribution mismatches without sharing raw data. This
          guarantees clean, compatible inputs and supports scientifically valid,
          fair collaborations.
        </span>
      </div>
      <img
        src="https://images.unsplash.com/photo-1707813130463-31d492f072ea?q=80&w=2131&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        alt="Data"
        className="hidden md:block h-60 lg:h-full w-full lg:w-[50%] object-cover object-center"
      />
    </div>,
    <div className="flex flex-col lg:flex-row w-screen h-[550px] md:h-[600px] shadow-md">
      <div className="text-[#020e16] h-full py-10 px-6 lg:py-20 lg:px-12 flex-1 overflow-hidden">
        <h1 className="text-[25px] md:text-[40px] lg:text-[50px] font-[800] mb-6 lg:mb-10">
          OMOP for Interoperability
        </h1>
        <span className="text-[16px] lg:text-[18px] font-[350] inline-block">
          True interoperability requires systems to not only share but also
          consistently interpret data across institutions. In federated
          learning, semantic and structural alignment is vital for meaningful,
          reproducible research. {NAME} promotes the OMOP Common Data Model,
          which standardizes observational health data using OHDSI vocabularies
          for semantic consistency. <br />
          <br />
          By supporting ETL pipelines and clinician-led data mapping, {
            NAME
          }{" "}
          helps institutions integrate local EHRs into a shared framework,
          reducing data wrangling and enabling efficient multi-center studies
          and reliable analyses.
        </span>
      </div>
      <img
        src="https://images.unsplash.com/photo-1477865300989-86ba6d4adcab?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        alt="Platform"
        className="hidden md:block h-60 lg:h-full w-full lg:w-[50%] object-cover object-center"
      />
    </div>,
    <div className="flex flex-col lg:flex-row w-screen h-[550px] md:h-[600px] shadow-md">
      <div className="text-[#020e16] h-full py-10 px-6 lg:py-20 lg:px-12 flex-1 overflow-hidden">
        <h1 className="text-[25px] md:text-[40px] lg:text-[50px] font-[800] mb-6 lg:mb-10">
          Training with NVFlare
        </h1>
        <span className="text-[16px] lg:text-[18px] font-[350] inline-block">
          The training of AI models is based on Federated Learning, a
          privacy-preserving approach where data remains securely within each
          institution. Models are trained locally, and only encrypted
          parameters—not raw patient data—are transmitted to a central server
          for aggregation. This method ensures compliance with data protection
          standards while enabling the creation of robust, consensus-based
          models from distributed clinical datasets.
          <br />
          <br />[
          <a
            href="https://blogs.nvidia.com/wp-content/uploads/2019/10/federated_learning_animation_still_white.png"
            className="text-[15px] lg:text-[17px] text-blue-700 px-1"
          >
            Source
          </a>
          ]
        </span>
      </div>
      <img
        src="https://blogs.nvidia.com/wp-content/uploads/2019/10/federated_learning_animation_still_white.png"
        alt="Federated Learning"
        className="hidden md:block h-60 lg:h-full w-full lg:w-[50%] object-contain object-center"
      />
    </div>,
    <div className="flex flex-col lg:flex-row w-screen h-[550px] md:h-[600px] shadow-md">
      <div className="text-[#020e16] h-full py-10 px-6 lg:py-20 lg:px-12 flex-1 overflow-hidden">
        <h1 className="text-[25px] md:text-[40px] lg:text-[50px] font-[800] mb-6 lg:mb-10">
          Inference
        </h1>
        <span className="text-[16px] lg:text-[18px] font-[350] inline-block">
          The platform provides an intuitive interface that allows healthcare
          professionals, even without technical expertise, to perform
          predictions in just a few clicks. Inference happens locally, directly
          within the hospital or research facility, ensuring that medical images
          and patient data never leave the institution. The only communication
          with the central system involves downloading model weights, preserving
          full control over patient privacy and data security.
        </span>
      </div>
      <img
        src={appHome}
        alt="Platform"
        className="hidden md:block h-60 lg:h-full w-full lg:w-[50%] object-cover object-center"
      />
    </div>,
  ];

  return (
    <div className="max-w-screen min-h-screen">
      <header
        className={`fixed top-0 left-0 w-screen z-50 transition-colors duration-300 ${
          scrolled ? "bg-[#020e16] shadow-md" : "bg-transparent"
        }`}
      >
        <div className="px-4 py-4">
          <img src={logo} alt="Logo" className="h-8 w-auto" />
        </div>
      </header>

      <div className="relative bg-[url(./assets/hero-2k.jpg)] ... h-screen bg-cover bg-center flex flex-col justify-center px-6 md:px-12 pt-30">
        <h1 className="text-[70px]/[1] md:text-[160px]/[1] font-[700] pb-8 md:pb-20">
          {NAME}
        </h1>
        <h1 className="text-[20px] md:text-[40px] font-[350] max-w-[900]">
          Infrastructure for Privacy Preserving Orchestration of Clinical
          Research for Trustworthy & Explainable AI
        </h1>
        <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 px-4 py-6">
          <img src={mouse} alt="Scroll" className="h-14 w-auto" />
        </div>
      </div>

      <div className="flex w-screen shadow-md py-14 md:py-30 bg-[#020e16] px-8 sm:px-15 md:px-20 lg:px-30">
        <span className="text-[16px] lg:text-[18px] font-[350] italic text-center">
          <strong>{NAME}</strong> is a pioneering collaborative technology
          platform set to transform clinical research through the power of
          federated learning. Designed specifically for the complexities of
          real-world medical environments, it enables healthcare institutions to
          securely train and deploy AI models across distributed data
          sources—without ever moving sensitive patient information. By
          orchestrating training directly within hospital infrastructures,
          {NAME} preserves data privacy, ensures regulatory compliance, and
          maintains data sovereignty. Its robust architecture supports secure
          communication, heterogeneous data formats, and adaptive algorithms
          that account for institutional and device-level variation.
          <br />
          <br />
          The platform includes customizable workflows, pre-trained model
          inference, and an intuitive user interface, making advanced AI tools
          readily accessible to clinicians and researchers. By bridging
          fragmented medical data and enabling privacy-preserving collaboration,
          {NAME} brings personalized, data-driven medicine closer to everyday
          clinical practice.
        </span>
      </div>

      <div className="relative bg-white">
        {slides[index]}
        <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 px-4 py-6 flex justify-center space-x-2 mt-4 ">
          {Array.from({ length: slides.length }).map((_, idx) => (
            <button
              key={idx}
              onClick={() => setIndex(idx)}
              className={`transition-all duration-300 rounded-full ${
                index === idx
                  ? "w-5 h-5 bg-blue-600 border-[#020e16] border-3"
                  : "w-3 h-3 my-1 bg-gray-400 border-[#020e16]"
              }`}
            />
          ))}
        </div>
      </div>

      <div className="overflow-x-auto py-10 md:py-25 shadow-md bg-[#020e16] mx-10 md:mx-20">
        <div className="flex gap-x-[100px]">
          <div className="min-w-70 md:min-w-130 lg:max-w-[300px] text-center">
            <p className="font-[800] text-[25px] lg:text-[30px] pb-5">Design (2021–2022)</p>
            <p className="font-[350] text-[16px] lg:text-[18px]">
              Defined technical architecture, user requirements, and core
              functionalities. Conducted feasibility studies, stakeholder
              consultations, and UX research to ensure scalable, secure, and
              user-centered platform design for long-term success.
            </p>
          </div>
          <div className="min-w-70 md:min-w-130 lg:max-w-[300px] text-center">
            <p className="font-[800] text-[25px] lg:text-[30px] pb-5">
              Implementation (2022–2024)
            </p>
            <p className="font-[350] text-[16px] lg:text-[18px]">
              Developed and tested platform features based on design
              specifications. Focused on backend integration, front-end
              development, performance optimization, and iterative testing to
              ensure functionality, stability, and cross-platform compatibility.
            </p>
          </div>
          <div className="min-w-70 md:min-w-130 lg:max-w-[300px] text-center">
            <p className="font-[800] text-[25px] lg:text-[30px] pb-5">
              Network Building (2025)
            </p>
            <p className="font-[350] text-[16px] lg:text-[18px]">
              Established robust network infrastructure, integrating secure
              protocols and scalable systems. Configured servers, services, and
              connectivity solutions to support platform operations, data flow,
              and future expansion needs.
            </p>
          </div>
        </div>
      </div>

      <div className="flex gap-x-[50px] w-screen shadow-md xl:flex-row flex-col bg-white">
        {/* https://www.mapchart.net/europe-nuts2.html, w-3000, h-2325 */}
        <img
          src={map}
          alt="Map"
          className="lg:h-250 xl:h-200 xl:px-12 rounded-2xl object-contain pt-0 sm:pt-20 pb-8 sm:pb-20"
          style={{ clipPath: "inset(2px 2px 2px 2px)" }}
        />
        <div className="text-[#020e16] h-full pt-0 pb-20 xl:pt-20 px-6 xl:px-0 xl:pr-12">
          <h1 className="text-[25px] md:text-[40px] lg:text-[50px] font-[800] mb-6 lg:mb-10">Become a Partner</h1>
          <div>
            <span className="text-[16px] lg:text-[18px] font-[350] inline-block">
              We are seeking institutional partners willing to contribute
              resources—such as funding, computational infrastructure, or domain
              expertise—to help advance the platform’s development. Your support
              will directly enhance our ability to build a scalable, secure, and
              clinically impactful ecosystem for AI-driven research. Together,
              we can shape the future of medical innovation.
              <br />
              <br />
              Ready to connect? Choose a button below to get started.
            </span>
            <div className="flex mt-10 sm:mt-20 gap-x-[20px]">
              <a
                href={`https://mail.google.com/mail/?ui=2&view=cm&fs=1&tf=1&to=${EMAIL}`}
                target="_blank"
                className="py-2 px-3 rounded-md border-[1px] border-gray-200"
              >
                <img src={gmail} alt="GMail" className="h-10 py-1" />
              </a>
              <a
                href={`mailto:${EMAIL}`}
                target="_blank"
                className="p-2 rounded-md border-[1px] border-gray-200"
              >
                <img src={outlook} alt="Outlook" className="h-10 py-0.5" />
              </a>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-[#020e16] shadow-md pt-20 pb-30 px-5 grid md:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="flex flex-col items-center text-center">
          <p className="text-[16px] mb-4">Main Partners</p>
          <div className="flex gap-4 flex-wrap justify-center bg-white py-2 px-4 rounded-md">
            <img src={iit} alt="IIT" className="h-12 md:h-14 py-2" />
            <img src={bracco} alt="Bracco" className="h-12 md:h-14 py-1" />
          </div>
        </div>
        <div className="flex flex-col items-center text-center">
          <p className="text-[16px] mb-4">Institutional Partners</p>
          <div className="flex gap-4 flex-wrap justify-center bg-white py-2 px-4 rounded-md">
            <img src={ucbm} alt="UCBM" className="h-12 md:h-14 py-2" />
          </div>
        </div>
        <div className="flex flex-col items-center text-center">
          <p className="text-[16px] mb-4">Scientific Partners</p>
          <div className="flex gap-4 flex-wrap justify-center bg-white py-2 px-4 rounded-md">
            <img src={orobix} alt="Orobix" className="h-12 md:h-14 py-5" />
            <img src={innomed} alt="Innomed" className="h-12 md:h-14 py-4" />
          </div>
        </div>
        <div className="flex flex-col items-center text-center">
          <p className="text-[16px] mb-4">Healthcare Partners</p>
          <div className="flex gap-4 flex-wrap justify-center bg-white py-2 px-4 rounded-md">
            <img src={gaslini} alt="Gaslini" className="h-12 md:h-14" />
            {/* <img src={sanmartino} alt="San Martino" className="h-12 md:h-14" /> */}
          </div>
        </div>
        <div className="flex flex-col items-center text-center">
          <p className="text-[16px] mb-4">Financial Partners</p>
          <div className="flex gap-4 flex-wrap justify-center bg-white py-2 px-4 rounded-md">
            <img src={mise} alt="MISE" className="h-12 md:h-14" />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
