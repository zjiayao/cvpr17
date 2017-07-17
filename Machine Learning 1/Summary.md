# Machine Learning 1


## [Spotlight 1-1A](./Machine%20Learning%201/Spotlight%201-1A)

### Exclusivity-Consistency Regularized Multi-View Subspace Clustering

* Abstract

Multi-view subspace clustering aims to partition a set of multi-source data into their underlying groups. To boost the performance of multi-view clustering, numerous subspace learning algorithms have been developed in recent years, but with rare exploitation of the representation complementarity between different views as well as the indicator conistency among the representations, let alone considering them simultaneously. In this paper, we propose a novel multi-view subspace clustering model that attempts to harness the complementary information between different repesentations by introducing a novel position-aware excluivity term. Meanwhile, a consistency term is employed to make these complementary representations to further have a common indicator. We formulate the above concerns ino a unified optimization framework. Experimental results on several benchmark datasets are conducted to reveal the effectiveness of our algorithm over other state-of-the-arts.

* Conclusion

In this paper, we have proposed a novel multi-view subspace clustering model namely ECMSC. Different from previous works, we simultaneously consider the complementary representation and consistent indicator into one framework. Moreover, a novel position-aware exclusivity term has been introduced to measure the diversity between different representations. In addition, an efficient alternative based algorithm has been proposed to seek the optimal solution. Extensive experimental results on several datasets have demonstrated the significant advantage of our method.


### Borrowing Treasures From the Wealthy: Deep Transfer Learning Through Selective Joint Fine-Tuning

* Code

[GitHub](https://github.com/ZYYSzj/Selective-Joint-Fine-tuning)

* Abstract

Deep neural networks require a large amount of labeled training data during supervised learning. However, collectng and labeling so much data might be infeasible in many cases. In this paper, we introduce a deep transfer learnng scheme, called selective joint fine-tuning, for improvng the performance of deep learning tasks with insufficient training data. In this scheme, a target learning task with insufficient training data is carried out simultaneously with another source learning task with abundant training data. However, the source learning task does not use all existing training data. Our core idea is to identify and use a subset of training images from the original source learning task whose low-level characteristics are similar to those from the target learning task, and jointly fine-tune shared conolutional layers for both tasks. Specifically, we compute descriptors from linear or nonlinear filter bank responses on training images from both tasks, and use such descripors to search for a desired subset of training samples for the source learning task.
Experiments demonstrate that our deep transfer learnng scheme achieves state-of-the-art performance on muliple visual classification tasks with insufficient training data for deep learning. Such tasks include Caltech 256, MIT Indoor 67, and fine-grained classification problems (Oxford Flowers 102 and Stanford Dogs 120). In comarison to fine-tuning without a source domain, the proosed method can improve the classification accuracy by 2% 10% using a single model

* Conclusion

In this paper, we address deep learning tasks with insuffiient training data by introducing a new deep transfer learnng scheme called selective joint fine-tuning, which perorms a target learning task with insufficient training data simultaneously with another source learning task with abunant training data. Different from previous work which diectly adds extra training data to the target learning task, our scheme borrows samples from a large-scale labeled dataset for the source learning task, and do not require addiional labeling effort beyond the existing datasets. Experients show that our deep transfer learning scheme achieves state-of-the-art performance on multiple visual classificaion tasks with insufficient training data for deep networks. **Nevertheless, how to find the most suitable source domain for a specific target learning task remains an open problem for future investigation.**

* Note

To find the most suitable source domain, one may refer to *Gong:2012:GFK*.

### The More You Know: Using Knowledge Graphs for Image Classification

* Abstract

One characteristic that sets humans apart from modern learning-based computer vision algorithms is the ability to acquire knowledge about the world and use that knowledge to reason about the visual world. Humans can learn about the characteristics of objects and the relationships that ocur between them to learn a large variety of visual conepts, often with few examples. This paper investigates the use of structured prior knowledge in the form of knowledge graphs and shows that using this knowledge improves perormance on image classification. We build on recent work on end-to-end learning on graphs, introducing the Graph Search Neural Network as a way of efficiently incorporating large knowledge graphs into a vision classification pipeline. We show in a number of experiments that our method outerforms standard neural network baselines for multi-label classification.

* Conclusion

In this paper, we present the Graph Search Neural Netork (GSNN) as a way of efficiently using knowledge graphs as extra information to improve image classificaion. We provide analysis that examines the flow of inforation through the GSNN and provides insights into why our model improves performance. We hope that this work provides a step towards bringing symbolic reasoning into traditional feed-forward computer vision frameworks.

The GSNN and the framework we use for vision probems is completely general. Our next steps will be to apply the GSNN to other vision tasks, such as detection, Visual Question Answering, and image captioning. Another intersting direction would be to combine the procedure of this work with a system such as NEIL [Chen:2013:Extracting Knowledge from Web Data] to create a system which builds knowledge graphs and then prunes them to get a more accurate, useful graph for image tasks.


### Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs

* Abstract

A number of problems can be formulated as prediction on graph-structured data. In this work, we generalize the convolution operator from regular grids to arbitrary graphs while avoiding the spectral domain, which allows us to hanle graphs of varying size and connectivity. To move beyond a simple diffusion, filter weights are conditioned on the speific edge labels in the neighborhood of a vertex. Together with the proper choice of graph coarsening, we explore contructing deep neural networks for graph classification. In particular, we demonstrate the generality of our formulaion in point cloud classification, where we set the new state of the art, and on a graph classification dataset, where we outperform other deep learning approaches.

* Conclusion

We have introduced edge-conditioned convolution (ECC), an operation on graph signal performed in the spaial domain where filter weights are conditioned on edge labels and dynamically generated for each specific input sample. We have shown that our formulation generalizes the standard convolution on graphs if edge labels are choen properly and experimentally validated this assertion on MNIST. We applied our approach to point cloud classifiation in a novel way, setting a new state of the art perormance on Sydney dataset. Furthermore, we have outerformed other deep learning-based approaches on graph classification dataset NCI1.

In feature work we would like to treat meshes as graphs rather than point clouds. Moreover, we plan to address the currently higher level of GPU memory consumption in case of large graphs with continuous edge labels, for example by randomized clustering, which could also serve as additional regularization through data augmentation.

### Convolutional Neural Network Architecture for Geometric Matching

* Abstract

We address the problem of determining correspondences between two images in agreement with a geometric model such as an affine or thin-plate spline transformation, and estimating its parameters. The contributions of this work are three-fold. First, we propose a convolutional neural netork architecture for geometric matching. The architecture is based on three main components that mimic the standard steps of feature extraction, matching and simultaneous inier detection and model parameter estimation, while being trainable end-to-end. Second, we demonstrate that the netork parameters can be trained from synthetically generted imagery without the need for manual annotation and that our matching layer significantly increases generalizaion capabilities to never seen before images. Finally, we show that the same model can perform both instance-level and category-level matching giving state-of-the-art results on the challenging Proposal Flow dataset.

* Conclusion

We have described a network architecture for geometic matching fully trainable from synthetic imagery withut the need for manual annotations. Thanks to our matchng layer, the network generalizes well to never seen beore imagery, reaching state-of-the-art results on the chalenging Proposal Flow dataset for category-level matching. This opens-up the possibility of applying our architecture to other difficult correspondence problems such as matchng across large changes in illumination (day/night) [Arandjelovic:2016:NetVLAD] or depiction style [Aubry:2013:Painting-to-3D].

### Deep Affordance-Grounded Sensorimotor Object Recognition

* Abstract

It is well-established by cognitive neuroscience that huan perception of objects constitutes a complex process, where object appearance information is combined with evience about the so-called object “affordances”, namely the types of actions that humans typically perform when ineracting with them. This fact has recently motivated the “sensorimotor” approach to the challenging task of autoatic object recognition, where both information sources are fused to improve robustness. In this work, the aforeentioned paradigm is adopted, surpassing current limitaions of sensorimotor object recognition research. Specifcally, the deep learning paradigm is introduced to the problem for the first time, developing a number of novel neuro-biologically and neuro-physiologically inspired arhitectures that utilize state-of-the-art neural networks for fusing the available information sources in multiple ways. The proposed methods are evaluated using a large RGB-D corpus, which is specifically collected for the task of sensoimotor object recognition and is made publicly available. Experimental results demonstrate the utility of affordance information to object recognition, achieving an up to 29% relative error reduction by its inclusion.

* Conlusion

In this paper, the problem of sensorimotor 3D obect recognition following the deep learning paradigm was investigated. A large public 3D object recogniion dataset was also introduced, including multiple obect types and a significant number of complex affordances, for boosting the research activities in the field. Two generalized neuro-biologically and neuro-physiologically grounded neural network architectures, implementing muliple fusion schemes for sensorimotor object recognition were presented and evaluated. The proposed sensorimoor multi-level slow fusion approach was experimentally shown to outperform similar probabilistic fusion methods of the literature. Future work will investigate the use of NN auto-encoders for modeling the human-object interacions in more details and the application of the proposed methodology to more realistic, “in-the-wild” object recogition data.

### Discovering Causal Signals in Images

* Abstract

The purpose of this paper is to point out and assay observable causal signals within collections of static images. We achieve this goal in two steps. First, we take a learning approach to observational causal inference, and build a classifier that achieves state-of-the-art performance on finding the causal direction between pairs of random variables, when given samples from their joint distribution. Second, we use our causal direction finder to effectively distinguish between features of obects and features of their contexts in collections of static images. Our experiments demonstrate the existence of (1) a relation between the direction of causality and the difference between objects and their contexts, and (2) observable causal sigals in collections of static images.

* Conclusion

Our experiments indicate the existence of statistically observable causal signals within sets of static images. However, further research is needed to best capture and exploit causal signals for applicaions in image understanding and robust object detection. In particular, we stress the importance of (1) building large, real-world datasets to aid research in causal inference, (2) extending data-driven techniques like NCC to causal inference of more than two variables, and (3) exploring data with explicit causal signals, such as the arrow of time in videos [Pickup:2014:Seeing the arrow of time].

### On Compressing Deep Models by Low Rank and Sparse Decomposition

## [Oral 1-1](./Machine%20Learning%201/Oral%201-1)

### PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

* Abstract

Point cloud is an important type of geometric data structure. Due to its irregular format, most researchers transform such data to regular 3D voxel grids or collections of images. This, however, renders data unnecessarily voluminous and causes issues. In this paper, we design a novel type of neural network that directly consumes point clouds, which well respects the permutation invariance of points in the input. Our network, named PointNet, provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing. Though simple, PointNet is highly efficient and effective. Empirically, it shows strong performance on par or even better than state of the art. Theoretically, we provide analysis towards understanding of what the network has learnt and why the network is robust with respect to input perturbation and corruption.

* Conclusion

In this work, we propose a novel deep neural network PointNet that directly consumes point cloud. Our network provides a unified approach to a number of 3D recognition tasks including object classification, part segmentation and semantic segmentation, while obtaining on par or better results than state of the arts on standard benchmarks. We also provide theoretical analysis and visualizations towards understanding of our network.

### Universal Adversarial Perturbations

* Code/Demo

[GitHub](https://github.com/LTS4/universal)

[YouTube](https://www.youtube.com/watch?v=jhOu5yhe0rc&feature=youtu.be)

* Abstract

Given a state-of-the-art deep neural network classifier, we show the existence of a universal (image-agnostic) and very small perturbation vector that causes natural images to be misclassified with high probability. We propose a systematic algorithm for computing universal perturbations, and show that state-of-the-art deep neural networks are highly vulnerable to such perturbations, albeit being quasiimperceptible to the human eye. We further empirically analyze these universal perturbations and show, in particular, that they generalize very well across neural networks. The surprising existence of universal perturbations reveals important geometric correlations among the high-dimensional decision boundary of classifiers. It further outlines potential security breaches with the existence of single directions in the input space that adversaries can possibly exploit to break a classifier on most natural images.

* Conclusion

We showed the existence of small universal perturbations that can fool state-of-the-art classifiers on natural images. We proposed an iterative algorithm to generate universal perturbations, and highlighted several properties of such perturbations. In particular, we showed that universal perturbations generalize well across different classification models, resulting in doubly-universal perturbations (imageagnostic, network-agnostic). We further explained the existence of such perturbations with the correlation between different regions of the decision boundary. This provides insights on the geometry of the decision boundaries of deep neural networks, and contributes to a better understanding of such systems. **A theoretical analysis of the geometric correlations between different parts of the decision boundary will be the subject of future research.**

### Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks

* Abstract

Collecting well-annotated image datasets to train modern machine learning algorithms is prohibitively expensive for many tasks. One appealing alternative is rendering synthetic data where ground-truth annotations are generated automatically. Unfortunately, models trained purely on rendered images often fail to generalize to real images. To address this shortcoming, prior work introduced unsupervised domain adaptation algorithms that attempt to map representations between the two domains or learn to extract features that are domain–invariant. In this work, we present a new approach that learns, in an unsupervised manner, a transformation in the pixel space from one domain to the other. Our generative adversarial network (GAN)–based method adapts source-domain images to appear as if drawn from the target domain. Our approach not only produces plausible samples, but also outperforms the state-of-the-art on a number of unsupervised domain adaptation scenarios by large margins. Finally, we demonstrate that the adaptation process generalizes to object classes unseen during training.

* Conclusion

We present a state-of-the-art method for performing unsupervised domain adaptation. Our models outperform previous work on a set of unsupervised domain adaptation scenarios, and in the case of the challenging “Synthetic Cropped Linemod to Cropped Linemod” scenario, our model more than halves the error for pose estimation compared to the previous best result. They are able to do so by using a GAN–based technique, stabilized by both a task-specific loss and a novel content–similarity loss. Furthermore, our model decouples the process of domain adaptation from the task-specific architecture, and provides the added benefit of being easy to understand via the visualization of the adapted image outputs of the model.

### Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

* Abstract

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN, a generative adversarial network (GAN) for image superresolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4× upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

* Conclusion

We have described a deep residual network SRResNet that sets a new state of the art on public benchmark datasets when evaluated with the widely used PSNR measure. We have highlighted some limitations of this PSNR-focused image super-resolution and introduced SRGAN, which augments the content loss function with an adversarial loss by training a GAN. Using extensive MOS testing, we have confirmed that SRGAN reconstructions for large upscaling factors (4×) are, by a considerable margin, more photo-realistic than reconstructions obtained with state-ofthe-art reference methods.
