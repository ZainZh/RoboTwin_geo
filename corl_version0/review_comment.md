Metareview:
This work proposes an object-centric continuous semantic field, a 3D representation built upon a tri-plane field conditioned on object point clouds and trained with PartNext part supervision. The core mechanism decouples support points sampled from raw observations from explicit 3D query locations, at which part-aware semantic embeddings are read out and fed to a DP3 policy. The authors claim this yields more stable functional-part cues under viewpoint and sampling variation, and report improved policy success over raw point-cloud, 2D feature-lifting, and 3D point-wise feature baselines. The method is evaluated on four RoboTwin simulation tasks and four real-world bimanual manipulation tasks.

Strengths:

The paper targets an important and timely problem: obtaining stable functional-part 3D representations for generalizable manipulation under real-world sensing variation.
The support/query separation is a clean and plausible design for reducing dependence on observation-specific point samples.
Real-robot improvements are substantial, and the controlled comparison using the same frozen encoder for the 3D point-wise baseline helps isolate the contribution of the proposed field.
The method is validated in both simulation and real bimanual tasks, demonstrating practical deployment value as a plug-in representation for DP3.
Weaknesses:

The technical novelty appears limited, as continuous neural fields, object-conditioned descriptor fields, and 3D semantic fields have been explored extensively in prior manipulation work.
It remains unclear how much of the improvement comes from the continuous field itself versus the PartNext part supervision or the clean point resampling, since the proposed training recipe is not ablated.
The validity of the 2D feature-lifting baseline is questionable: its implementation is not described, and the reported results appear inconsistent with findings from D3Fields and F3RM, making it difficult to judge whether the motivation stems from a fundamental limitation or a weak baseline.
The evaluation relies mainly on downstream success rates and lacks direct representation-level metrics such as part mIoU, feature consistency, or cross-instance correspondence accuracy.
The reliance on category-level PartNext annotations and per-category field training limits scalability and sits uneasily with the generalizability claimed in the title.
Pre-Rebuttal Recommendation: First-round reject: All reviewers and the AC scored below weak accept. The paper should not proceed to rebuttal.


review1:

Summary:
This paper observes that existing 3D representations is not consistent across sampled observations. Therefore, this paper proposes an object-centric continuous semantic field. Authors show that the proposed representation could provide more stable functional cues and policy improvement.

Strengths:
The comparison to the prior works (Figure 1) is very clear and informative.
The representation visualization and analysis are intuitive.
Weaknesses:
Supplementary video quality can be improved. Current video is so blurry that I cannot read clearly. Also, there seems to be no voiceover. Readers cannot easily understand the video content without proper voiceover.
Figure 2 is quite dense and quite challenging for readers to digest.
Paper organization can be further improved. Current method description is very detailed, while the experiment description is very brief and a lot of important details are missing. For example, authors do not explain how they implement 2D lifting baselines.
Missing baseline details: this could be my major complaint. One of the key motivations of this work is that 2D feature lifting is not sufficient and consistent. However, according to work D3Fields or F3RM [23, 24], the extracted feature can be very consistent and also part-aware, similar to what the authors show in the paper. But results shown in the paper are not consistent with findings and results from D3Fields and F3RM. Since authors do not descrive how they implemented 2D feature lifting in the paper, it is a bit hard to judge whether the bad performance of the baseline comes from a bad implementation or a fundamental flaw of the method. If it is only due to a bad implementation, the motivation of this work might be less solid.
Performance gap between proposed method and the baselines is much larger in the real world than the one in the simulation. More analysis on this is appreciated.
Questions For Authors:
In L116, authors mention that "During policy learning and deployment, support points are sampled from the observed object point cloud". Does it assume that the point cloud can be fully observed?
What is g_phi in Equation 2? Is that the adapter? What is the purpose of the adapter (e.g. mapping semantic features into a lower space)? How is the adapter built?
Is d_u the same as d_s in L135?
In Figure 2, the trapezoids at the bottom of "1. Field Construction" section seem to imply that XY Plane feature & ZY Plane feature can be used to derive XZ Plane feature. However, the text does not imply so. A clarification is appreciated.
Limitations And Broader Impact:
Limitation is well discussed in the paper.
Overall Score: 4: Borderline. The paper has interesting ideas but notable weaknesses — e.g., insufficient experiments, unclear contribution, or limited novelty. It is unlikely authors could address all the issues in the limited period of the rebuttal.
Confidence Score: 4: High confidence. I am knowledgeable in this area and confident in my assessment.
Ethical Concerns: None
LLM Disclosure: No
Official Review of Submission2063 by Reviewer rLBP
Official Reviewby Reviewer rLBP13 Jul 2026, 10:09 (modified: 05 Aug 2026, 05:34)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, AuthorsRevisions
Summary:
This paper presents a pipeline for 3D robotic manipulation that aims to make point-cloud policies (specifically DP3) more robust to sensor noise and varying viewpoints. The core idea is simple but effective: instead of attaching semantic features directly to the messy, raw point clouds captured by the cameras, the authors use those raw points to condition a continuous 3D field (using tri-planes). They then sample a fresh, clean set of 3D query points from that field, evaluate their part-aware embeddings, and feed these clean points into the robot's policy. The field is trained heavily on PartNext annotations. The authors test this approach on 4 simulation tasks and 4 real-world bimanual tasks, showing solid improvements over standard point-cloud and 2D-lifting baselines.

Strengths:
The core premise makes a lot of sense. Decoupling the "support" points (what the camera happens to see) from the "query" points (where the policy actually reads the features) is a very smart way to handle real-world sensor instability.
The performance jump on the real robot is substantial (e.g., improving mug grasping from 7/20 to 17/20). It is clear that this representation genuinely helps the policy handle real-world variations, partial observations, and geometry changes.
I appreciate that the learned field can be frozen and plugged straight into DP3 as an extra modality without messing with the action representations. Furthermore, the experimental controls are good—using the same frozen Utonia encoder for the 3D point-wise baseline really helps isolate the benefits of the proposed field.
Weaknesses:
My biggest hesitation with this paper is its technical novelty. Continuous neural fields, object-conditioned descriptor fields (like NDFs), and 3D semantic fields have all been explored extensively in manipulation. This paper feels more like a very well-executed systems engineering effort—stitching together Utonia, tri-planes, PartNext supervision, and DP3—rather than a fundamental algorithmic breakthrough.
The authors introduce a specific training recipe (part anchoring, cross-instance alignment, augmentation stability) but don't ablate it. Right now, it is entirely unclear where the performance gains are coming from. Is it the continuous field itself? The heavy part-level supervision? Or just the fact that you are resampling clean points?
Training a separate field for every single object category using dense PartNext annotations feels a bit like a step backward given the community's recent shift toward open-vocabulary or self-supervised 3D features. This heavy supervision requirement limits how scalable and "generalizable" (as claimed in the title) this method truly is.
Questions For Authors:
Please provide an ablation (at least in simulation) showing performance when: Removing each of the three training losses, querying the features only at the original observed points (no resampling), using predicted part probabilities instead of the learned continuous embeddings.
Based on those ablations, is the performance gain actually coming from the continuous field, or is it mostly coming from having strong, canonical PartNext supervision?
How exactly does this approach differ technically from the concurrent part-aware 3D feature-field work cited, or prior continuous descriptor fields? A direct empirical comparison to one of these would significantly strengthen the paper.
Limitations And Broader Impact:
N/A

Overall Score: 4: Borderline. The paper has interesting ideas but notable weaknesses — e.g., insufficient experiments, unclear contribution, or limited novelty. It is unlikely authors could address all the issues in the limited period of the rebuttal.
Confidence Score: 4: High confidence. I am knowledgeable in this area and confident in my assessment.
Ethical Concerns: None
Ethical Concerns Details:
N/A

LLM Disclosure: Yes — please describe below
LLM Disclosure Details:
I have asked LLM to fix grammar and refine the sentence I wrote to more easy read ways. But the opinions are from my own and I am responsible for all the reviewing contents.


Review3:

The paper proposes an object-centric continuous semantic field that maps an object point cloud and 3D query locations to part-aware semantic embeddings for manipulation. It trains the field using PartNext part labels, cross-instance part alignment, and augmentation consistency, then freezes it to generate semantic point clouds as observations for a DP3 imitation-learning policy. Experiments in RoboTwin and real-world bimanual tasks show higher success rates than raw point clouds, 2D feature lifting, and 3D point-wise feature baselines.

Strengths:
The paper tackles an important problem in robot learning: obtaining stable functional-part representations for generalizable manipulation.

The proposed support/query separation is technically clean and provides a plausible way to reduce dependence on view- and sample-specific point features.

The method is validated in both simulation and real-world bimanual manipulation tasks, with consistent improvements over raw point cloud, 2D lifting, and 3D point-wise feature baselines.

Weaknesses:
The paper makes representation-level claims but mostly evaluates with downstream success rates. It should include direct metrics such as part mIoU, feature consistency, or cross-instance correspondence accuracy.

The comparisons are limited. Nearby methods such as feature-SDF/semantic implicit fields, 3DGS feature fields, and stronger continuous 3D feature baselines should be discussed or compared.

The method depends on category-level part labels from PartNext, which limits applicability to objects with clear canonical parts.

The ablation study is insufficient; it should isolate the effects of support/query separation, part alignment, and augmentation consistency.

Questions For Authors:
Can the authors provide direct representation metrics, such as part mIoU, cross-instance correspondence accuracy, or feature consistency under viewpoint/sampling perturbations?

How much of the improvement comes from support/query separation versus the use of PartNext part supervision? A point-wise part-supervised baseline would help clarify this.

Why are nearby continuous 3D feature representations, such as feature-SDF/semantic implicit fields or 3DGS-based feature fields, not discussed or compared?

How sensitive is the method to noisy or partial object point clouds, especially under single-view observations or imperfect segmentation?

What happens when objects have ambiguous, continuous, or task-dependent functional regions rather than canonical part labels?

Limitations And Broader Impact:
The method relies on category-level part annotations and consistent part taxonomies, which may limit its applicability to objects with ambiguous, deformable, or task-dependent functional regions. It also assumes reasonably accurate object point clouds, so performance may degrade under severe occlusion, poor segmentation, or single-view partial observations. Broader impact is mostly positive for improving robot manipulation generalization, but deployment in real-world settings should consider safety risks from perception errors or incorrect functional-part predictions.

Overall Score: 4: Borderline. The paper has interesting ideas but notable weaknesses — e.g., insufficient experiments, unclear contribution, or limited novelty. It is unlikely authors could address all the issues in the limited period of the rebuttal.
Confidence Score: 3: Moderate confidence. I am familiar with the area but not an expert; some aspects may be outside my expertise.
Ethical Concerns: None
LLM Disclosure: Yes — please describe below
LLM Disclosure Details:
Yes. I used ChatGPT/Codex to help understand the paper, identify related concerns, and polish the review text.
