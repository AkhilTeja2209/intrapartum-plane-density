"""Content for the BCSE497J Project-I report.

Kept separate from build_report.py so the prose can be edited without touching
the document-manipulation logic. Every number here comes from docs/RESULTS.md.

Each CONTENT entry is a list of (kind, text):
    body  normal justified paragraph
    item  bold-lead list item ("Name: description")
    h3    third-level heading
    cap   centred caption line
"""

COVER = {
    "title": ("DEEP LEARNING-BASED CLASSIFICATION OF STANDARD AND "
              "NON-STANDARD INTRAPARTUM ULTRASOUND IMAGES"),
    "reg": "23BCB0135",
    "name": "MORISETTY VENKATA SAI AKHIL TEJA",
    "guide": "Dr. Mythili. T",
    "guide_desig": "<<Designation - please fill in>>",
    "specialization": "(Specialization: Bioinformatics)",
    "month_year": "September 2026",
}

ABSTRACT = [
    ("body",
     "Transperineal ultrasound is used during labour to measure the angle of "
     "progression and the head-symphysis distance, but those measurements are "
     "only valid on a standard plane, a frame in which the pubic symphysis and "
     "the fetal skull contour are both clearly resolved. Selecting that frame "
     "from a long sweep is operator-dependent and slow. This project builds a "
     "deep learning classifier that labels an intrapartum ultrasound frame as "
     "standard or non-standard, and uses it to answer a methodological "
     "question the literature has left open."),
    ("body",
     "Published work reports that models trained on dense video frames "
     "outperform models trained on curated still images, and reads this as "
     "evidence that video data is intrinsically richer. That inference is "
     "unsupported, because the dense arm also has far more training samples. "
     "This project separates the two by constructing both arms from a single "
     "corpus, the IUGC 2024 intrapartum dataset of 774 videos, varying only "
     "the number of frames drawn per video so that probe, anatomy, scanner, "
     "annotator and label definition are held constant by construction. A "
     "matched-budget arm then caps the dense arm to the sparse arm's exact "
     "frame count, and a prior-matched arm additionally fixes the class "
     "balance."),
    ("body",
     "A ResNet-18 was trained under thirteen sampling conditions and a "
     "ResNet-18 with a bidirectional LSTM head was trained for the temporal "
     "arm, all evaluated on the same held-out 8,665-frame official test set "
     "with video-level bootstrap confidence intervals. Frame density produced "
     "no measurable benefit: the strongest single result came from 434 "
     "training frames, one per video, against 53,996 for the dense arm. "
     "Repeating the headline conditions over three seeds showed that both "
     "matched-budget comparisons reverse sign between seeds, with one "
     "condition varying by 0.23 macro-F1 on identical data. The principal "
     "finding is therefore a measurement result: at this corpus size, "
     "seed-to-seed variance exceeds any density effect, so the single-run "
     "comparisons on which the prior claim rests cannot support it. The "
     "trained classifier is deployed as a browser-based demonstrator."),
    ("body",
     "Keywords - Intrapartum ultrasound, Standard plane classification, "
     "Convolutional neural network, Frame sampling density, Temporal "
     "modelling, Reproducibility."),
]

INTRO = [
    ("body",
     "Ultrasound is the primary imaging modality in obstetrics because it is "
     "non-ionising, portable and inexpensive. During the second stage of "
     "labour, transperineal ultrasound is used to assess fetal head descent "
     "objectively, replacing a digital examination whose reproducibility "
     "between examiners is poor. The two established measurements are the "
     "angle of progression, taken between the long axis of the pubic "
     "symphysis and a line tangent to the fetal skull, and the "
     "head-symphysis distance."),
    ("body",
     "Both measurements are defined only on a standard plane: a mid-sagittal "
     "view in which the pubic symphysis is visible along its full long axis "
     "and the fetal skull contour is resolved. A sweep contains many frames, "
     "of which only a contiguous subset satisfies that condition. Identifying "
     "them is currently a manual, operator-dependent step performed under "
     "time pressure in a delivery room."),
    ("body",
     "Automating that step is a binary frame classification problem, and "
     "convolutional neural networks have been applied to closely related "
     "fetal standard plane tasks with good results. This project trains such "
     "a classifier on the IUGC 2024 intrapartum corpus and, in doing so, "
     "examines a methodological assumption that recurs throughout this "
     "literature: that training on densely sampled video frames is superior "
     "to training on sparsely curated images."),
]

MOTIVATION = [
    ("body",
     "Two things motivate this project. The first is clinical. A tool that "
     "proposes the single most measurable frame in a sweep converts a "
     "subjective, repeated judgement into one reviewable suggestion, which is "
     "the form in which such a system would actually be deployed."),
    ("body",
     "The second is methodological, and it is what makes the project a study "
     "rather than an implementation exercise. Papers comparing image-trained "
     "and video-trained models routinely change the dataset and the sample "
     "count at the same time, then attribute the resulting difference to the "
     "richness of video. The comparison cannot distinguish that explanation "
     "from the simpler one that the video arm had more data. No published "
     "work isolates the two."),
    ("body",
     "The isolation is achievable. Drawing k frames per video from a single "
     "video corpus produces a genuine synthetic image dataset in which every "
     "confound, probe, gestational stage, anatomy, scanner, annotator and "
     "label definition, is constant by construction, and k becomes a "
     "continuous dial from image dataset to video dataset. Capping the dense "
     "arm to the sparse arm's exact frame count then decides whether any "
     "advantage came from density or from sample count. That measurement is "
     "cheap, has not been made, and is what this project contributes."),
]

SCOPE = [
    ("body",
     "The project covers binary standard-plane classification on the IUGC "
     "2024 intrapartum transperineal corpus: 774 videos, from which 65,531 "
     "frames were extracted, distributed under CC-BY-4.0. The label space is "
     "standard versus non-standard. Segmentation of the pubic symphysis and "
     "fetal head, and direct regression of the angle of progression, are "
     "outside the scope, although the corpus supplies the annotations that "
     "would support them."),
    ("body",
     "The architecture is deliberately held fixed. A single ImageNet-"
     "initialised ResNet-18 encoder is used throughout, with one classifier "
     "head for the frame-wise arm and a bidirectional LSTM head for the "
     "temporal arm. Optimiser, learning rate, schedule, augmentation, "
     "early-stopping rule and class weighting are identical in every "
     "condition. Architecture search and hyper-parameter tuning are out of "
     "scope, because varying them would reintroduce the confound the study "
     "exists to remove."),
    ("body",
     "Evaluation is on the dataset's own held-out test split, scored with "
     "imbalance-aware metrics and confidence intervals bootstrapped over "
     "videos rather than frames. The deliverables are the trained classifier, "
     "a browser-based demonstrator, and the measured comparison between "
     "sparse and dense sampling. Clinical validation and any form of "
     "deployment on patients are explicitly outside the scope."),
]

LITREV = [
    ("body",
     "Automatic standard plane detection in fetal ultrasound is an "
     "established problem. Chen et al. showed that features transferred from "
     "natural-image networks localise fetal standard planes far better than "
     "hand-crafted descriptors, establishing transfer learning as the default "
     "for this domain. Baumgartner et al. extended this with SonoNet, which "
     "detects and weakly localises thirteen fetal standard planes in freehand "
     "sweeps in real time using only image-level labels, and remains the "
     "reference architecture for the task."),
    ("body",
     "Burgos-Artizzu et al. released FETAL_PLANES_DB, a curated corpus of "
     "roughly 12,400 maternal-fetal images across six anatomical plane "
     "classes, and benchmarked several convolutional architectures on it. "
     "That dataset is the canonical example of the curated still-image "
     "regime: one representative frame per acquisition, chosen by a "
     "sonographer. It is frequently used as the image-side baseline against "
     "which video-trained models are compared."),
    ("body",
     "For the intrapartum setting specifically, ISUOG practice guidelines "
     "define the transperineal views and the measurement protocol, and work "
     "by Kalache et al. established the angle of progression as a predictor "
     "of delivery mode. These define what a standard plane means clinically "
     "and therefore what the classifier's positive class actually is."),
    ("body",
     "On the modelling side, the backbone used here is the residual network "
     "of He et al., initialised from ImageNet, and the temporal head is the "
     "long short-term memory unit of Hochreiter and Schmidhuber. Optimisation "
     "follows decoupled weight decay. For validating that a classifier "
     "attends to anatomy rather than to acquisition artefacts, Grad-CAM "
     "provides class-discriminative localisation without architectural "
     "change."),
    ("body",
     "Across this literature two patterns recur. First, comparisons between "
     "image-trained and video-trained models change the corpus and the sample "
     "count together. Second, temporal architectures are benchmarked against "
     "unsmoothed frame-wise baselines, even though standard planes occur in "
     "contiguous runs and a zero-parameter moving average over frame-wise "
     "probabilities recovers much of that structure without any temporal "
     "model. Both patterns are addressed directly by the design in Section "
     "2.3."),
    ("body",
     "Note on citation count: the template permits up to fifty journal "
     "papers. The list in Section 5 is deliberately short and contains only "
     "references that have been verified. It should be expanded during "
     "Project-II rather than padded."),
]

GAP = [
    ("body",
     "Gap 1 - Sampling density is never isolated from sample count. Studies "
     "reporting that video-derived training data outperforms curated images "
     "vary the corpus, the label space, the anatomy and the number of "
     "training samples simultaneously. Any difference observed is therefore "
     "uninterpretable: it may be density, sample count, or task difficulty. "
     "No prior work reports a budget-matched comparison in which both arms "
     "receive the same number of frames."),
    ("body",
     "Gap 2 - Temporal models are benchmarked against the wrong baseline. "
     "Because standard planes occur in contiguous runs, most exploitable "
     "temporal structure is label autocorrelation, which post-hoc smoothing "
     "of frame-wise probabilities captures with no parameters and no "
     "training. Comparing a recurrent model against an unsmoothed frame-wise "
     "model therefore attributes to the architecture a gain that "
     "post-processing already provides."),
    ("body",
     "Gap 3 - Run-to-run variance is not reported. Results on corpora of a "
     "few hundred patients are typically reported from a single training run. "
     "If seed-to-seed variance is comparable to the effect being measured, "
     "such a comparison cannot support its conclusion. This project measures "
     "that variance directly, and the measurement turns out to govern the "
     "interpretation of everything else."),
]

OBJECTIVES = [
    ("item",
     "Objective 1: To construct a reproducible, verified frame-level dataset "
     "from the IUGC 2024 intrapartum corpus, with labels joined at a rate of "
     "1.0 and every data-integrity invariant enforced as an assertion."),
    ("item",
     "Objective 2: To train a ResNet-18 classifier that distinguishes "
     "standard from non-standard intrapartum ultrasound frames, evaluated on "
     "the official held-out test split with imbalance-aware metrics."),
    ("item",
     "Objective 3: To quantify the effect of frame sampling density by "
     "training the identical architecture under thirteen sampling conditions "
     "spanning one frame per video to every frame."),
    ("item",
     "Objective 4: To decide whether any density effect is attributable to "
     "density or to sample count, using a budget-matched arm and a "
     "prior-matched arm that additionally fixes the class balance."),
    ("item",
     "Objective 5: To evaluate whether explicit temporal modelling improves "
     "on a frame-wise model with tuned post-hoc smoothing, rather than on an "
     "unsmoothed baseline."),
    ("item",
     "Objective 6: To quantify seed-to-seed variance across repeated runs and "
     "report every comparison against it."),
    ("item",
     "Objective 7: To deploy the trained classifier as a browser-based "
     "demonstrator that performs inference locally, with the exported model "
     "verified against the training-time model."),
]

PROBLEM = [
    ("body",
     "Given a single frame from an intrapartum transperineal ultrasound "
     "sweep, decide whether it is a standard plane, that is, whether the "
     "pubic symphysis and the fetal skull contour are both resolved well "
     "enough for the angle of progression to be measured. Formally, learn a "
     "function that maps a greyscale frame to a probability, thresholded at a "
     "value selected on validation data and frozen before the test split is "
     "used."),
    ("body",
     "The problem is made non-trivial by three properties of the corpus. The "
     "classes are close to balanced at the frame level but the useful "
     "positives occur in contiguous runs, so frames are not independent and "
     "any split must be performed at video level to avoid scoring a model on "
     "near-duplicates of its own training data. The training videos are "
     "trimmed clips in which the plane is held throughout, while validation "
     "and test videos are untrimmed sweeps that pass in and out of plane, so "
     "there is a genuine distribution shift between training and evaluation. "
     "And the corpus spans three hospitals and several scanner models, so a "
     "model can score well by recognising acquisition characteristics rather "
     "than anatomy."),
    ("body",
     "The associated research question is whether the density with which "
     "training frames are drawn from video changes what the classifier "
     "learns, once sample count and class prior are held constant."),
]

PLAN = [
    ("body",
     "The project is planned over sixteen weeks. Weeks one to twelve, shown "
     "in blue in Fig. 1, are complete as of Review 2 and cover the literature "
     "survey, dataset acquisition and integrity audit, frame extraction and "
     "index construction, split design, both experimental arms, the repeat-"
     "seed variance analysis and the browser deployment. Weeks thirteen to "
     "sixteen, shown in green, remain: the Grad-CAM anatomical attention "
     "check, the paired video bootstrap and the final report."),
    ("fig", "fig1_gantt.png|6.2|Fig. 1. Project plan (Gantt chart)"),
]

FUNCTIONAL = [
    ("item", "Frame extraction: The system shall decode each ultrasound video "
             "once into pre-resized greyscale frames, writing through a "
             "Unicode-safe path so that no video is silently skipped."),
    ("item", "Label parsing: The system shall parse per-video label files, "
             "including the ALL and NONE sentinels that denote whole-video "
             "labels, and shall fail loudly rather than silently producing an "
             "empty label set."),
    ("item", "Index construction: The system shall produce a single "
             "frame-level index recording frame path, video identifier, frame "
             "position, label and originating split, and shall refuse to "
             "write it if the label join rate is below 1.0."),
    ("item", "Video-level splitting: The system shall partition data by video "
             "identifier, never by frame, and shall assert that no video "
             "appears in more than one split."),
    ("item", "Sampling conditions: The system shall build a training set for "
             "any condition specified as frames per video, temporal stride, "
             "total frame budget, or target class prior."),
    ("item", "Model training: The system shall train the frame-wise and "
             "temporal arms through a single trainer, so that both receive an "
             "identical optimiser, schedule, augmentation and stopping rule."),
    ("item", "Evaluation: The system shall report balanced accuracy, "
             "macro-F1, AUPRC and MCC on the held-out test set, with "
             "confidence intervals bootstrapped over videos, and shall select "
             "the decision threshold on validation data only."),
    ("item", "Inference service: The system shall accept a single ultrasound "
             "frame and return the probability that it is a standard plane "
             "together with the decision at the frozen threshold."),
]

NONFUNCTIONAL = [
    ("item", "Reproducibility: Every run shall be seeded, and every reported "
             "comparison shall be repeated across at least three seeds so "
             "that run-to-run variance is measurable."),
    ("item", "Verifiability: Data-integrity properties shall be enforced as "
             "assertions with explicit override flags, not as log messages, "
             "because a silent failure in this pipeline produces plausible "
             "numbers rather than an error."),
    ("item", "Comparability: All conditions shall be scored on the identical "
             "complete test set, and no condition shall receive a tuning "
             "budget the others do not."),
    ("item", "Performance: Inference on a single frame shall complete within "
             "one second on commodity client hardware without a GPU."),
    ("item", "Privacy: The deployed demonstrator shall perform inference "
             "locally, so that no ultrasound image is transmitted or stored."),
    ("item", "Portability: The trained model shall be exportable to an "
             "interoperable format and shall reproduce the training-time "
             "model's outputs to within a verified numerical tolerance."),
    ("item", "Maintainability: The pipeline shall be modular, and a synthetic "
             "end-to-end test shall run in continuous integration on every "
             "change."),
    ("item", "Safety: Any public interface shall state prominently that the "
             "system is a research artefact and is not validated for clinical "
             "use."),
]

FEASIBILITY = [
    ("h3", "3.2.1 Technical Feasibility"),
    ("body",
     "The full study has been executed on a single laptop-class GPU. Thirteen "
     "sampling conditions, two temporal variants and two additional seeds of "
     "the headline conditions, twenty-five training runs in total, completed "
     "in approximately five GPU-hours on an NVIDIA RTX 3070 Ti with 8 GB of "
     "memory. The dataset is publicly available under CC-BY-4.0, and the "
     "software stack is entirely open source. No component of the project "
     "depends on hardware or data that is not already in hand."),
    ("h3", "3.2.2 Economic Feasibility"),
    ("body",
     "Direct cost is effectively zero. The corpus is openly licensed, all "
     "libraries are open source, computation used existing hardware, and the "
     "demonstrator is hosted on a free static-hosting service. The economic "
     "argument for the work is that it substitutes a cheap, decisive "
     "measurement, whether density or sample count explains a reported "
     "effect, for the expensive alternative of collecting more video data on "
     "the assumption that density is what matters."),
    ("h3", "3.2.3 Social Feasibility"),
    ("body",
     "The intended use is assistive: the system proposes a frame for a "
     "clinician to accept or reject, and does not make a diagnosis. This "
     "keeps the clinician in the decision loop and makes the output "
     "reviewable. The corpus is de-identified and openly licensed, so no "
     "additional consent or data-protection process is required for the "
     "research use made of it here."),
    ("body",
     "Two risks are recognised and handled explicitly. The corpus originates "
     "from three hospitals, so a model may generalise poorly to other "
     "populations and scanners; the planned Grad-CAM attention check and a "
     "leave-one-centre-out split are intended to detect this. And any "
     "publicly reachable interface can be mistaken for a clinical tool, so "
     "the deployed demonstrator states on the page that it is a research "
     "artefact, is not a medical device, and returns a confident number even "
     "for images unlike anything in its training distribution."),
]

HARDWARE = [
    ("item", "Processor: AMD/Intel x86-64, 16 logical cores."),
    ("item", "Memory (RAM): 16 GB."),
    ("item", "Storage: 10 GB free, holding the 1.1 GB source archive and "
             "approximately 3 GB of extracted frames."),
    ("item", "Graphics Processing Unit (GPU): NVIDIA GeForce RTX 3070 Ti "
             "Laptop GPU, 8 GB VRAM, CUDA 12.4."),
    ("item", "Client hardware for the demonstrator: any device with a modern "
             "browser; no GPU required."),
]

SOFTWARE = [
    ("item", "Operating System: Windows 11; the pipeline is platform "
             "independent and is tested on Linux in continuous integration."),
    ("item", "Programming Languages: Python 3.13 for the pipeline, JavaScript "
             "for the browser demonstrator."),
    ("item", "Development Environment: Git for version control, GitHub "
             "Actions for continuous integration and deployment."),
    ("item", "Libraries and Frameworks: PyTorch 2.6.0 with CUDA 12.4, "
             "TorchVision, NumPy, pandas, scikit-learn, OpenCV, Pillow, "
             "Matplotlib, ONNX and ONNX Runtime, onnxruntime-web."),
    ("item", "Data storage: the frame index and split definitions are held as "
             "CSV and JSON; no database server is required."),
    ("item", "Deployment: GitHub Pages serving a static page that runs the "
             "exported model client-side."),
]

ARCHITECTURE = [
    ("body",
     "The system is a linear pipeline with a single branch point at the model "
     "head, shown in Fig. 2. Videos, label files and segmentation "
     "annotations enter at the top. Frame extraction decodes each video once "
     "to pre-resized JPEG frames, which avoids re-decoding the same video "
     "hundreds of times across conditions and epochs, and writes through a "
     "Unicode-safe path. In parallel, a sentinel-aware parser reads the label "
     "files, expanding the whole-video ALL and NONE markers against each "
     "video's frame count."),
    ("body",
     "The two streams meet at a single frame-level index of 65,531 rows, "
     "which every downstream stage reads. Making this the one source of truth "
     "is deliberate: if each experiment re-derived its own labels, conditions "
     "would differ in more than the variable under study. The index is "
     "written only if the label join rate is exactly 1.0 for every split."),
    ("body",
     "Below the index, the split is fixed once at video level and reused by "
     "every condition, so no condition is scored on a different test set. The "
     "sampling module then produces each condition's training set, and the "
     "splicing module synthesises label transitions for the temporal arm. "
     "The two arms share one encoder definition and differ only in the head: "
     "the frame-wise arm emits one probability per frame, and the temporal "
     "arm emits one probability per frame of a clip, so both produce output "
     "of the same shape and the same metrics apply without re-derivation."),
    ("body",
     "The evaluation stage applies the frozen threshold, computes "
     "imbalance-aware metrics with video-level bootstrap intervals, and "
     "optionally applies post-hoc smoothing to the frame-wise arm to form the "
     "baseline the temporal arm must beat. The trained frame-wise model is "
     "exported to ONNX for the browser demonstrator."),
    ("fig", "fig2_architecture.png|6.2|Fig. 2. System architecture"),
]

DESIGN_INTRO = [
    ("body",
     "The design is documented with a data flow diagram, which shows how data "
     "moves between processes and stores, and a use case diagram, which shows "
     "what each actor can do with the system."),
]

DFD = [
    ("body",
     "Fig. 3 shows the level-1 data flow. Four processes operate over four "
     "data stores. Process 1.0 extracts frames from the video corpus (D1) "
     "into the frame store (D2). Process 2.0 joins those frames to the label "
     "files and writes the index and split definitions (D3). Process 3.0 "
     "trains and evaluates a model under one sampling condition, reading the "
     "index and writing the model and its metrics (D4). Process 4.0 is the "
     "inference path used by the demonstrator: it takes a query frame from "
     "the user, applies the trained model, and returns a probability and a "
     "decision. The researcher drives processes 1.0 to 3.0; the clinician "
     "interacts only with 4.0."),
    ("fig", "fig3_dfd.png|6.2|Fig. 3. Data flow diagram (level 1)"),
]

USECASE = [
    ("body",
     "Fig. 4 shows the actors and their use cases. The researcher prepares "
     "the dataset and builds the index, configures a sampling condition, "
     "trains a model and evaluates it on the held-out test set. The clinician "
     "submits an ultrasound frame and views the resulting plane decision and "
     "probability; viewing the decision is included in submitting a frame, "
     "since the demonstrator returns both in one step. Separating the actors "
     "reflects the deployment boundary: dataset preparation and training are "
     "offline research activities, while classification is the only "
     "user-facing operation."),
    ("fig", "fig4_usecase.png|5.7|Fig. 4. Use case diagram"),
]

IMPLEMENTATION = [
    ("body",
     "Implementation is complete for both experimental arms. The dataset "
     "pipeline, the thirteen Arm 1 sampling conditions, the Arm 2 temporal "
     "model with its splicing ablation, the three-seed repetition of the "
     "headline conditions and the browser deployment are all finished, "
     "amounting to twenty-five training runs. The Grad-CAM anatomical "
     "attention check and the paired video bootstrap remain, and are "
     "scheduled in Fig. 1 for the weeks after this review."),
    ("h3", "4.3.1 Dataset preparation and audit"),
    ("body",
     "The audit stage proved more consequential than expected. Four defects "
     "were found in data preparation, none of which raised an error, and "
     "together they had left the index with zero labelled training frames. "
     "The label file marks whole-video positives with the string ALL, which "
     "is not a numeric list and so parsed to an empty set; 266 of the 434 "
     "training videos are named with a doubled filename in the published "
     "archive and therefore matched no label row; the frame writer failed "
     "silently on 71 videos whose filenames contain non-ASCII characters, "
     "returning a failure code that was not checked; and two label files "
     "encoding the same frames were deduplicated without first verifying that "
     "they agreed. After correction the index contains 65,531 labelled frames "
     "over 774 videos with a label join rate of 1.0000 on every split. These "
     "invariants are now enforced as assertions."),
    ("body",
     "The audit also established two facts that changed the experimental "
     "design. The official test labels are public in this release, so the "
     "dataset's own split can be used rather than carving a held-out set from "
     "the training data. And the training split consists of trimmed clips in "
     "which the plane is held from the first frame to the last, containing "
     "zero label transitions per video, whereas validation and test videos "
     "are sweeps containing 0.93 and 1.35 transitions per video respectively. "
     "This is a genuine train-test distribution shift and it constrains what "
     "the temporal arm can learn."),
    ("h3", "4.3.2 Arm 1: sampling density"),
    ("body",
     "Thirteen conditions were trained, spanning 434 training frames at one "
     "frame per video to 53,996 frames at every frame, plus budget-matched "
     "and prior-matched dense arms. No density effect was observed. The "
     "sparse arm ranged from 0.588 to 0.687 macro-F1 across a twentyfold "
     "range of training frames with no trend, and the dense arm ranged from "
     "0.506 to 0.637 across an eightfold range, also with no trend. The "
     "strongest single result in the study, 0.6872 macro-F1, came from the "
     "smallest training set of 434 frames. The dense arm did not outperform "
     "the sparse arm even in the unmatched comparison that prior work reports "
     "as a win for video data."),
    ("h3", "4.3.3 Arm 2: temporal modelling"),
    ("body",
     "Because the training split contains no label transitions, a recurrent "
     "model trained on it can reach zero training loss by ignoring time "
     "entirely. Transitions were therefore synthesised by splicing a "
     "contiguous run from a positive video to one from a negative video, "
     "drawn from the same acquisition session where one exists. On the same "
     "dense condition, the frame-wise model reached 0.5785 macro-F1 raw and "
     "0.5453 with tuned post-hoc smoothing, while the bidirectional LSTM "
     "reached 0.6695 with splicing enabled. The temporal arm therefore beat "
     "the smoothed baseline, which is the opposite of what was anticipated. "
     "However, the ablation with splicing disabled reached 0.6496 with "
     "heavily overlapping intervals, and that model cannot have learned "
     "transitions, so the temporal advantage is not attributable to "
     "transition modelling."),
    ("h3", "4.3.4 Variance across seeds"),
    ("body",
     "At a single seed the dense arm lost every budget-matched comparison, "
     "which read as a clean result. Repeating the headline conditions at two "
     "further seeds reversed it. The budget-matched difference moved from "
     "-0.106 to +0.013 to -0.093 across seeds, and the prior-matched "
     "difference from -0.168 to +0.121 to -0.045; both comparisons change "
     "sign and both standard deviations exceed their means. One condition "
     "varied by 0.231 macro-F1 between seeds on identical data with an "
     "identical recipe."),
    ("body",
     "The principal finding of the project follows from this. On a corpus of "
     "434 training videos with this train-test distribution shift, "
     "seed-to-seed variance reaches roughly plus or minus 0.12 macro-F1, "
     "which is larger than any sampling-density effect present. A single-run "
     "comparison between sparse and dense training at this scale therefore "
     "cannot distinguish a real effect from a change of random seed. Since "
     "the prior work this project set out to examine reports exactly such "
     "single-run comparisons, this is a stronger and more directly relevant "
     "result than a positive finding would have been."),
    ("h3", "4.3.5 Deployment"),
    ("body",
     "The trained frame-wise model is exported to ONNX and served as a static "
     "page that performs inference in the browser, so no image leaves the "
     "user's device. Two verification steps guard the export, because both "
     "possible failures are silent: the exported graph is compared against "
     "the training-time model on 256 real test frames and rejected if the "
     "logits differ by more than a fixed tolerance, and the JavaScript "
     "preprocessing reproduces the training-time resize geometry exactly "
     "rather than relying on the browser's own image scaling, which was "
     "measured to shift the output probability by 0.041 on a real frame. "
     "After correction the browser reproduces the reference probabilities to "
     "within seven parts in a million."),
]

REFERENCES = [
    ("body", "Journals: <IEEE Format>"),
    ("body",
     "[1] C. F. Baumgartner, K. Kamnitsas, J. Matthew, T. P. Fletcher, S. "
     "Smith, L. M. Koch, B. Kainz, and D. Rueckert, \"SonoNet: Real-time "
     "detection and localisation of fetal standard scan planes in freehand "
     "ultrasound,\" IEEE Transactions on Medical Imaging, vol. 36, no. 11, "
     "pp. 2204-2215, 2017."),
    ("body",
     "[2] H. Chen, D. Ni, J. Qin, S. Li, X. Yang, T. Wang, and P. A. Heng, "
     "\"Standard plane localization in fetal ultrasound via domain "
     "transferred deep neural networks,\" IEEE Journal of Biomedical and "
     "Health Informatics, vol. 19, no. 5, pp. 1627-1636, 2015."),
    ("body",
     "[3] X. P. Burgos-Artizzu, D. Coronado-Gutierrez, B. Valenzuela-Alcaraz, "
     "E. Bonet-Carne, E. Eixarch, F. Crispi, and E. Gratacos, \"Evaluation of "
     "deep convolutional neural networks for automatic classification of "
     "common maternal fetal ultrasound planes,\" Scientific Reports, vol. 10, "
     "no. 1, p. 10200, 2020."),
    ("body",
     "[4] S. Hochreiter and J. Schmidhuber, \"Long short-term memory,\" "
     "Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997."),
    ("body",
     "[5] A. Kalache, J. Duckelmann, V. Michaelis, J. Lange, G. Cichon, and "
     "K. Dudenhausen, \"Transperineal ultrasound imaging in prolonged second "
     "stage of labor with occipitoanterior presenting fetuses: how well does "
     "the angle of progression predict the mode of delivery?,\" Ultrasound in "
     "Obstetrics and Gynecology, vol. 33, no. 3, pp. 326-330, 2009."),
    ("body",
     "[6] T. Ghi, T. Eggebo, C. Lees, K. Kalache, P. Rozenberg, A. Youssef, "
     "L. J. Salomon, and B. Tutschek, \"ISUOG Practice Guidelines: intrapartum "
     "ultrasound,\" Ultrasound in Obstetrics and Gynecology, vol. 52, no. 1, "
     "pp. 128-139, 2018."),
    ("body", "Conference: <IEEE Format>"),
    ("body",
     "[7] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning for "
     "image recognition,\" in Proc. IEEE Conf. Computer Vision and Pattern "
     "Recognition (CVPR), 2016, pp. 770-778."),
    ("body",
     "[8] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and "
     "D. Batra, \"Grad-CAM: Visual explanations from deep networks via "
     "gradient-based localization,\" in Proc. IEEE Int. Conf. Computer Vision "
     "(ICCV), 2017, pp. 618-626."),
    ("body",
     "[9] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "
     "\"ImageNet: A large-scale hierarchical image database,\" in Proc. IEEE "
     "Conf. Computer Vision and Pattern Recognition (CVPR), 2009, "
     "pp. 248-255."),
    ("body",
     "[10] I. Loshchilov and F. Hutter, \"Decoupled weight decay "
     "regularization,\" in Proc. Int. Conf. Learning Representations (ICLR), "
     "2019."),
    ("body",
     "[11] O. Ronneberger, P. Fischer, and T. Brox, \"U-Net: Convolutional "
     "networks for biomedical image segmentation,\" in Proc. Medical Image "
     "Computing and Computer-Assisted Intervention (MICCAI), 2015, "
     "pp. 234-241."),
    ("body", "Datasets and Weblinks:"),
    ("body",
     "[12] IUGC 2024 Intrapartum Ultrasound Grand Challenge dataset "
     "(DatasetV3), Zenodo, doi: 10.5281/zenodo.17655183, CC-BY-4.0. "
     "[Online]. Available: https://zenodo.org/records/17655183"),
    ("body",
     "[13] Project source code, experimental logs and browser demonstrator. "
     "[Online]. Available: "
     "https://github.com/AkhilTeja2209/intrapartum-plane-density"),
]

CONTENT = {
    "ABSTRACT": ABSTRACT,
    "1. INTRODUCTION": [],
    "1.1 Background": INTRO,
    "1.2 Motivation": MOTIVATION,
    "1.3 Scope of the Project": SCOPE,
    "2. PROJECT DESCRIPTION AND GOALS": [],
    "2.1 Literature Review": LITREV,
    "2.2 Research Gap": GAP,
    "2.3 Objectives": OBJECTIVES,
    "2.4 Problem Statement": PROBLEM,
    "2.5 Project Plan": PLAN,
    "3. TECHNICAL SPECIFICATION": [],
    "3.1 Requirements": [],
    "3.1.1": FUNCTIONAL,
    "3.1.2": NONFUNCTIONAL,
    "3.2 Feasibility Study": FEASIBILITY,
    "3.3 System Specification": [],
    "3.3.1 Hardware Specification": HARDWARE,
    "3.3.2 Software Specification": SOFTWARE,
    "4. DESIGN APPROACH AND DETAILS": [],
    "4.1 System Architecture": ARCHITECTURE,
    "4.2 Design": DESIGN_INTRO,
    "4.2.1 Data Flow Diagram": DFD,
    "4.2.2 Use Case Diagram": USECASE,
    "4.3": IMPLEMENTATION,
    "5. REFERENCES": REFERENCES,
}

TOC_ROWS = [
    ("", "Abstract", "i"),
    ("1.", "INTRODUCTION", "1"),
    ("", "1.1 Background", "1"),
    ("", "1.2 Motivation", "1"),
    ("", "1.3 Scope of the Project", "2"),
    ("2.", "PROJECT DESCRIPTION AND GOALS", "3"),
    ("", "2.1 Literature Review", "3"),
    ("", "2.2 Research Gap", "4"),
    ("", "2.3 Objectives", "4"),
    ("", "2.4 Problem Statement", "5"),
    ("", "2.5 Project Plan", "6"),
    ("3.", "TECHNICAL SPECIFICATION", "7"),
    ("", "3.1 Requirements", "7"),
    ("", "3.1.1 Functional", "7"),
    ("", "3.1.2 Non-Functional", "8"),
    ("", "3.2 Feasibility Study", "9"),
    ("", "3.3 System Specification", "10"),
    ("4.", "DESIGN APPROACH AND DETAILS", "11"),
    ("", "4.1 System Architecture", "11"),
    ("", "4.2 Design", "12"),
    ("", "4.3 Implementation Status and Results", "14"),
    ("5.", "REFERENCES", "18"),
]
