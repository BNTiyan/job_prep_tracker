// AI/ML Engineer Interview Preparation Plan (60 days)
// Tailored for Google AI/ML roles based on your background
// Includes Cloud Services (AWS & GCP) for MLOps proficiency
const preparationPlan = [
    // Week 1: ML Fundamentals Review & Python for ML
    {
        day: 1,
        category: "Machine Learning Fundamentals",
        tasks: [
            {
                title: "Supervised Learning Review: Linear & Logistic Regression",
                duration: 50,
                notes: "Mathematics behind gradient descent, cost functions. Implement from scratch in NumPy."
            },
            {
                title: "Bias-Variance Tradeoff & Regularization (L1/L2)",
                duration: 30,
                notes: "When to use Ridge vs Lasso. Impact on model complexity."
            },
            {
                title: "Coding: Implement Linear Regression from Scratch",
                duration: 20,
                notes: "No sklearn - use only NumPy. Understand matrix operations."
            },
            {
                title: "☁️ AWS Fundamentals: IAM & EC2 Basics",
                duration: 20,
                notes: "IAM roles, policies, users. Launch EC2 instance. Security groups. Cost optimization basics."
            }
        ]
    },
    {
        day: 2,
        category: "Machine Learning Fundamentals",
        tasks: [
            {
                title: "Decision Trees & Random Forests",
                duration: 50,
                notes: "Gini impurity, entropy, information gain. Feature importance. Overfitting prevention."
            },
            {
                title: "Ensemble Methods: Bagging, Boosting, Stacking",
                duration: 30,
                notes: "XGBoost, LightGBM, AdaBoost internals. When to use each."
            },
            {
                title: "Coding: Implement Decision Tree Classifier",
                duration: 20,
                notes: "Recursive splitting algorithm. Practice on Iris dataset."
            },
            {
                title: "☁️ AWS Lambda & Serverless",
                duration: 20,
                notes: "Lambda functions basics. Triggers (S3, API Gateway). Use cases for ML inference. Cold start optimization."
            }
        ]
    },
    {
        day: 3,
        category: "Deep Learning Fundamentals",
        tasks: [
            {
                title: "Neural Networks Basics: Forward & Backpropagation",
                duration: 50,
                notes: "Chain rule, activation functions (ReLU, Sigmoid, Tanh). Vanishing gradient problem."
            },
            {
                title: "Optimization Algorithms: SGD, Adam, RMSProp",
                duration: 30,
                notes: "Learning rate scheduling. Momentum and adaptive learning rates."
            },
            {
                title: "Implement: Simple Neural Network in NumPy",
                duration: 20,
                notes: "2-layer network for XOR problem. Manual backprop calculation."
            },
            {
                title: "☁️ AWS S3 & Data Storage",
                duration: 20,
                notes: "S3 buckets, versioning, lifecycle policies. Data lakes. S3 Select. Glacier for archiving. Access patterns for ML datasets."
            }
        ]
    },
    {
        day: 4,
        category: "Deep Learning - CNNs",
        tasks: [
            {
                title: "Convolutional Neural Networks Architecture",
                duration: 50,
                notes: "Conv layers, pooling, stride, padding. Parameter calculation. ResNet, VGG, Inception."
            },
            {
                title: "Transfer Learning & Fine-tuning",
                duration: 30,
                notes: "When to freeze layers. Feature extraction vs fine-tuning. ImageNet pretrained models."
            },
            {
                title: "Code: Fine-tune ResNet on Custom Dataset (PyTorch)",
                duration: 20,
                notes: "Use your ADAS camera object detection experience. Practice explaining CV pipeline."
            },
            {
                title: "☁️ AWS Step Functions & Orchestration",
                duration: 20,
                notes: "State machines for ML workflows. Error handling, retry logic. Integrate with Lambda, SageMaker. ETL pipeline orchestration."
            }
        ]
    },
    {
        day: 5,
        category: "Computer Vision Advanced",
        tasks: [
            {
                title: "Object Detection: YOLO, R-CNN Family, SSD",
                duration: 50,
                notes: "Anchor boxes, NMS, IoU. Two-stage vs one-stage detectors. Real-time inference."
            },
            {
                title: "2D to 3D Mapping & Depth Estimation",
                duration: 30,
                notes: "Stereo vision, monocular depth. Relate to your Continental ADAS work."
            },
            {
                title: "Review Week 1: Practice Explaining Your CV Projects",
                duration: 20,
                notes: "Continental ADAS camera detection, Kalman filters, 3D mapping - prepare STAR stories."
            },
            {
                title: "☁️ GCP Fundamentals: Cloud Storage & Compute Engine",
                duration: 20,
                notes: "GCS buckets, object storage classes. Compute Engine vs App Engine. IAM basics. BigQuery intro."
            }
        ]
    },

    // Week 2: Deep Learning & NLP
    {
        day: 6,
        category: "Deep Learning - RNNs & LSTMs",
        tasks: [
            {
                title: "Recurrent Neural Networks & Sequential Data",
                duration: 50,
                notes: "LSTM, GRU architecture. Vanishing gradient in RNNs. Bidirectional RNNs."
            },
            {
                title: "Sequence-to-Sequence Models & Attention",
                duration: 30,
                notes: "Encoder-decoder architecture. Attention mechanism fundamentals."
            },
            {
                title: "Code: Implement LSTM for Time Series Prediction",
                duration: 20,
                notes: "PyTorch LSTM. Stock price or sensor data prediction."
            },
            {
                title: "☁️ AWS Glue & ETL Pipelines",
                duration: 20,
                notes: "Glue jobs, crawlers, Data Catalog. ETL transformations. S3 to Redshift. PySpark on Glue."
            }
        ]
    },
    {
        day: 7,
        category: "Natural Language Processing",
        tasks: [
            {
                title: "Transformers Architecture Deep Dive",
                duration: 50,
                notes: "Self-attention, multi-head attention, positional encoding. 'Attention is All You Need' paper."
            },
            {
                title: "BERT, GPT, T5 - Pretrained Models",
                duration: 30,
                notes: "Masked LM vs autoregressive. Fine-tuning strategies. Your Vertex AI (Gemini) experience."
            },
            {
                title: "🏗️ System Design: URL Shortener (bit.ly)",
                duration: 30,
                notes: "Design components: Hashing, database sharding, rate limiting, caching. Draw architecture diagram. Scalability considerations."
            },
            {
                title: "☁️ AWS SageMaker Basics",
                duration: 10,
                notes: "Training jobs, endpoints, model registry. Managed notebooks. Integration with S3."
            }
        ]
    },
    {
        day: 8,
        category: "NLP & LLMs",
        tasks: [
            {
                title: "Large Language Models: Architecture & Training",
                duration: 60,
                notes: "GPT-3/4 architecture. Prompt engineering. Few-shot learning. Relate to your Beacon AI project."
            },
            {
                title: "Retrieval-Augmented Generation (RAG)",
                duration: 40,
                notes: "Vector databases (Pinecone, ChromaDB). Embeddings. Context injection. Your SAST platform experience."
            },
            {
                title: "Code: Build Simple RAG System with OpenAI API",
                duration: 20,
                notes: "Document Q&A with embeddings. Practice prompt orchestration (Beacon experience)."
            }
        ]
    },
    {
        day: 9,
        category: "MLOps & Production ML",
        tasks: [
            {
                title: "ML Pipeline Design: Training, Serving, Monitoring",
                duration: 45,
                notes: "Feature stores, model registry. Versioning. Online vs offline inference. CI/CD for ML."
            },
            {
                title: "MLOps Tools: MLflow, Kubeflow, Vertex AI, SageMaker",
                duration: 35,
                notes: "Experiment tracking. Model deployment. A/B testing. Leverage your Databricks + Bosch experience."
            },
            {
                title: "Review: Your Rivian Databricks Pipelines",
                duration: 20,
                notes: "Prepare to discuss: ML-driven reviewer recommendations, anomaly detection, feedback loops."
            },
            {
                title: "☁️ AWS SageMaker Pipelines & Model Registry",
                duration: 20,
                notes: "SageMaker Pipelines for ML workflows. Model registry for versioning. Endpoint auto-scaling. Model monitoring."
            }
        ]
    },
    {
        day: 10,
        category: "Behavioral - ML Projects",
        tasks: [
            {
                title: "Prepare STAR Stories for ML Projects",
                duration: 40,
                notes: "Beacon AI SAST, Databricks dashboards, ADAS object detection, computational trust research."
            },
            {
                title: "Google's AI Principles & Responsible AI",
                duration: 30,
                notes: "Fairness, explainability, safety. Your AI governance experience at Rivian."
            },
            {
                title: "Practice: 'Walk me through your ML project' (5 min)",
                duration: 20,
                notes: "Record yourself. Problem → Approach → Results → Impact."
            },
            {
                title: "🏗️ System Design: Rate Limiter & API Gateway",
                duration: 30,
                notes: "Token bucket vs leaky bucket algorithms. Distributed rate limiting with Redis. API throttling strategies."
            }
        ]
    },

    // Week 3: ML Systems & Algorithms
    {
        day: 11,
        category: "ML System Design",
        tasks: [
            {
                title: "Design a Recommendation System (YouTube/Netflix)",
                duration: 70,
                notes: "Collaborative filtering, content-based, hybrid. Cold start problem. Real-time vs batch."
            },
            {
                title: "Feature Engineering & Selection",
                duration: 30,
                notes: "Feature importance, dimensionality reduction (PCA). Feature crosses."
            },
            {
                title: "Read: Google's Recommendation System Paper",
                duration: 20,
                notes: "YouTube recommendation architecture. Candidate generation vs ranking."
            }
        ]
    },
    {
        day: 12,
        category: "ML System Design",
        tasks: [
            {
                title: "Design a Search Ranking System",
                duration: 70,
                notes: "Learning to rank (LTR). NDCG, MAP metrics. Two-stage ranking. Personalization."
            },
            {
                title: "A/B Testing & Experimentation",
                duration: 30,
                notes: "Statistical significance, p-values. Multi-armed bandits. Your KPI dashboard experience."
            },
            {
                title: "Code: Implement Simple Collaborative Filtering",
                duration: 20,
                notes: "Matrix factorization with NumPy. MovieLens dataset."
            }
        ]
    },
    {
        day: 13,
        category: "ML Algorithms - Unsupervised",
        tasks: [
            {
                title: "Clustering: K-Means, DBSCAN, Hierarchical",
                duration: 50,
                notes: "Elbow method, silhouette score. Density-based vs centroid-based."
            },
            {
                title: "Dimensionality Reduction: PCA, t-SNE, UMAP",
                duration: 50,
                notes: "Eigenvalues/eigenvectors. Visualization techniques. When to use each."
            },
            {
                title: "Anomaly Detection Algorithms",
                duration: 20,
                notes: "Isolation Forest, One-Class SVM. Relate to your anomaly detection work at Rivian."
            }
        ]
    },
    {
        day: 14,
        category: "ML Theory & Math",
        tasks: [
            {
                title: "Probability & Statistics Refresher",
                duration: 60,
                notes: "Bayes theorem, conditional probability, distributions (Normal, Poisson, Binomial)."
            },
            {
                title: "Linear Algebra for ML",
                duration: 40,
                notes: "Matrix operations, eigendecomposition, SVD. Why important for ML."
            },
            {
                title: "Information Theory: Entropy, KL Divergence",
                duration: 20,
                notes: "Cross-entropy loss. Information gain in decision trees."
            }
        ]
    },
    {
        day: 15,
        category: "ML System Design",
        tasks: [
            {
                title: "Design an Ad Click Prediction System",
                duration: 70,
                notes: "Logistic regression at scale. Feature engineering. Real-time serving. CTR prediction."
            },
            {
                title: "Model Serving & Inference Optimization",
                duration: 30,
                notes: "TensorFlow Serving, TorchServe. Model quantization, pruning. Your AWS Lambda experience."
            },
            {
                title: "Week 3 Review & Reflections",
                duration: 20,
                notes: "Update progress tracker. Identify weak areas."
            }
        ]
    },

    // Week 4: Advanced ML & Reinforcement Learning
    {
        day: 16,
        category: "Reinforcement Learning",
        tasks: [
            {
                title: "RL Fundamentals: MDP, Bellman Equations",
                duration: 60,
                notes: "States, actions, rewards, policy. Value functions. Discount factor."
            },
            {
                title: "Q-Learning & Deep Q-Networks (DQN)",
                duration: 40,
                notes: "Experience replay, target networks. Atari game playing."
            },
            {
                title: "Code: Implement Q-Learning for GridWorld",
                duration: 20,
                notes: "OpenAI Gym environment. Visualize learned policy."
            }
        ]
    },
    {
        day: 17,
        category: "Reinforcement Learning",
        tasks: [
            {
                title: "Policy Gradient Methods: REINFORCE, A3C, PPO",
                duration: 60,
                notes: "Actor-critic architecture. Proximal Policy Optimization. Your computational trust research."
            },
            {
                title: "RL Applications: Robotics, Autonomous Driving",
                duration: 40,
                notes: "Your ADAS experience. Safety constraints. Simulation to real-world transfer."
            },
            {
                title: "Read: AlphaGo Paper & DeepMind Blog",
                duration: 20,
                notes: "Monte Carlo Tree Search + Deep RL. Google's RL applications."
            }
        ]
    },
    {
        day: 18,
        category: "ML Coding Interview",
        tasks: [
            {
                title: "Implement K-Nearest Neighbors from Scratch",
                duration: 40,
                notes: "Distance metrics, KD-trees for efficiency. Classification vs regression."
            },
            {
                title: "Implement K-Means Clustering from Scratch",
                duration: 40,
                notes: "Lloyd's algorithm. Initialization strategies. Convergence criteria."
            },
            {
                title: "Implement Gradient Descent Variants",
                duration: 40,
                notes: "Batch, Stochastic, Mini-batch. Learning rate scheduling."
            }
        ]
    },
    {
        day: 19,
        category: "ML System Design",
        tasks: [
            {
                title: "Design a Fraud Detection System",
                duration: 70,
                notes: "Class imbalance, SMOTE. Real-time scoring. Feature engineering for fraud."
            },
            {
                title: "Handling Imbalanced Datasets",
                duration: 30,
                notes: "Undersampling, oversampling, class weights. Precision-recall tradeoff."
            },
            {
                title: "Evaluation Metrics Deep Dive",
                duration: 20,
                notes: "Precision, Recall, F1, AUC-ROC, AUC-PR. When to use each. Your security work context."
            }
        ]
    },
    {
        day: 20,
        category: "Mock Interview - ML",
        tasks: [
            {
                title: "ML System Design Mock Interview",
                duration: 60,
                notes: "Design recommendation system or search ranking. Practice on Pramp or with peer."
            },
            {
                title: "ML Coding Round Mock",
                duration: 40,
                notes: "Implement ML algorithm from scratch. Explain time/space complexity."
            },
            {
                title: "Month 1 Review: Assess Progress",
                duration: 20,
                notes: "What concepts need more work? Adjust plan for weeks 5-8."
            }
        ]
    },

    // Week 5: Advanced Topics & Specialization
    {
        day: 21,
        category: "Graph Neural Networks",
        tasks: [
            {
                title: "GNN Fundamentals: Message Passing, Aggregation",
                duration: 60,
                notes: "Graph convolutions. Node embeddings. GCN, GAT, GraphSAGE architectures."
            },
            {
                title: "Applications: Social Networks, Knowledge Graphs",
                duration: 40,
                notes: "Node classification, link prediction, graph classification."
            },
            {
                title: "Code: Implement GCN with PyTorch Geometric",
                duration: 20,
                notes: "Cora dataset. Node classification task."
            }
        ]
    },
    {
        day: 22,
        category: "Generative AI",
        tasks: [
            {
                title: "Generative Adversarial Networks (GANs)",
                duration: 60,
                notes: "Generator vs discriminator. Mode collapse. DCGAN, StyleGAN. Training stability."
            },
            {
                title: "Variational Autoencoders (VAEs)",
                duration: 40,
                notes: "Latent space, reparameterization trick. VAE vs GAN tradeoffs."
            },
            {
                title: "Diffusion Models Basics",
                duration: 20,
                notes: "DALL-E, Stable Diffusion. Forward/reverse diffusion process."
            }
        ]
    },
    {
        day: 23,
        category: "Model Optimization & Deployment",
        tasks: [
            {
                title: "Model Compression: Quantization, Pruning, Distillation",
                duration: 60,
                notes: "INT8 quantization. Knowledge distillation from teacher to student model."
            },
            {
                title: "Edge Deployment: TensorFlow Lite, ONNX",
                duration: 40,
                notes: "Mobile/embedded ML. Your automotive experience. Latency constraints."
            },
            {
                title: "GPU Optimization & Distributed Training",
                duration: 20,
                notes: "Data parallelism, model parallelism. Multi-GPU training strategies."
            }
        ]
    },
    {
        day: 24,
        category: "ML Security & Safety",
        tasks: [
            {
                title: "Adversarial Machine Learning",
                duration: 60,
                notes: "FGSM, PGD attacks. Adversarial training. Your cybersecurity background."
            },
            {
                title: "Model Robustness & Uncertainty Quantification",
                duration: 40,
                notes: "Calibration, prediction intervals. Monte Carlo dropout. Safety-critical ML."
            },
            {
                title: "Review: Your Beacon AI SAST Project",
                duration: 20,
                notes: "AI-assisted security findings. Prompt orchestration. False positive reduction."
            }
        ]
    },
    {
        day: 25,
        category: "Time Series & Forecasting",
        tasks: [
            {
                title: "Time Series Analysis: ARIMA, Prophet",
                duration: 60,
                notes: "Stationarity, autocorrelation, seasonality. Traditional vs ML approaches."
            },
            {
                title: "Deep Learning for Time Series: LSTM, Temporal CNNs",
                duration: 40,
                notes: "Sequence modeling. Multi-horizon forecasting. Attention for time series."
            },
            {
                title: "Code: Build Time Series Forecasting Model",
                duration: 20,
                notes: "Stock prices or IoT sensor data. Your automotive telemetry experience."
            }
        ]
    },

    // Week 6: Data Engineering for ML & Scalability
    {
        day: 26,
        category: "Data Engineering for ML",
        tasks: [
            {
                title: "Feature Stores: Architecture & Best Practices",
                duration: 60,
                notes: "Feast, Tecton. Online vs offline features. Feature versioning. Your Databricks work."
            },
            {
                title: "Data Pipelines: ETL for ML",
                duration: 40,
                notes: "Airflow, Prefect. Your AWS Lambda/S3 experience. Data quality checks."
            },
            {
                title: "Data Versioning: DVC, Delta Lake",
                duration: 20,
                notes: "Reproducibility. Data lineage. Your Snowflake analytics experience."
            }
        ]
    },
    {
        day: 27,
        category: "Distributed ML & MLOps Infrastructure",
        tasks: [
            {
                title: "Distributed Training: Horovod, PyTorch DDP, Ray",
                duration: 45,
                notes: "Parameter servers, ring-allreduce. Gradient synchronization. Multi-node training."
            },
            {
                title: "Spark for ML: MLlib, PySpark",
                duration: 30,
                notes: "Large-scale data processing. Distributed feature engineering. Delta Lake for ML."
            },
            {
                title: "Code: Train Model on Multi-GPU Setup",
                duration: 20,
                notes: "PyTorch DistributedDataParallel. Measure speedup."
            },
            {
                title: "☁️ Kubernetes for ML: KubeFlow, Seldon Core",
                duration: 25,
                notes: "K8s fundamentals for ML. KubeFlow pipelines. Model serving with Seldon. Auto-scaling pods."
            }
        ]
    },
    {
        day: 28,
        category: "ML Model Monitoring",
        tasks: [
            {
                title: "Model Drift Detection & Monitoring",
                duration: 60,
                notes: "Data drift vs concept drift. Statistical tests. Alerting strategies."
            },
            {
                title: "Observability for ML Systems",
                duration: 40,
                notes: "Metrics, logging, tracing. Prometheus, Grafana. Your dashboard building experience."
            },
            {
                title: "Continuous Training & Online Learning",
                duration: 20,
                notes: "When to retrain. Incremental learning. Model versioning strategies."
            }
        ]
    },
    {
        day: 29,
        category: "ML System Design - Advanced",
        tasks: [
            {
                title: "Design a Real-time Personalization System",
                duration: 70,
                notes: "Context-aware recommendations. Cold start. Low latency serving. Your AUTOSIM experience."
            },
            {
                title: "Design an Autonomous Driving Perception System",
                duration: 50,
                notes: "Multi-sensor fusion. Your ADAS camera work. Real-time constraints. Safety requirements."
            }
        ]
    },
    {
        day: 30,
        category: "Mock Interview - Advanced ML",
        tasks: [
            {
                title: "Full ML Interview Loop Simulation",
                duration: 90,
                notes: "ML coding + ML system design back-to-back. Record yourself."
            },
            {
                title: "Behavioral: Leadership & Impact Stories",
                duration: 30,
                notes: "Your product owner experience. Cross-functional collaboration. Rivian team leadership."
            }
        ]
    },

    // Week 7: Google-Specific Preparation
    {
        day: 31,
        category: "Google ML Infrastructure",
        tasks: [
            {
                title: "Google's ML Stack: TensorFlow, JAX, Vertex AI",
                duration: 60,
                notes: "TPUs vs GPUs. TensorFlow Extended (TFX). Your Vertex AI (Gemini) experience."
            },
            {
                title: "Google Research Papers Review",
                duration: 40,
                notes: "Attention is All You Need, BERT, EfficientNet, MobileNet. Understand innovations."
            },
            {
                title: "Google AI Products: Deep Dive",
                duration: 20,
                notes: "Google Search (RankBrain), Translate, Photos, Assistant. ML behind products."
            }
        ]
    },
    {
        day: 32,
        category: "Google ML System Design",
        tasks: [
            {
                title: "Design Google Photos Search",
                duration: 70,
                notes: "Image embeddings, vector search, face recognition. Privacy considerations."
            },
            {
                title: "Design Google Translate",
                duration: 50,
                notes: "Neural machine translation. Multilingual models. Zero-shot translation."
            }
        ]
    },
    {
        day: 33,
        category: "ML Algorithms - Advanced",
        tasks: [
            {
                title: "Bayesian Machine Learning",
                duration: 60,
                notes: "Bayesian inference, MCMC, variational inference. Gaussian processes."
            },
            {
                title: "Meta-Learning & Few-Shot Learning",
                duration: 40,
                notes: "MAML, Prototypical networks. Learning to learn. Transfer learning advanced."
            },
            {
                title: "Multi-Task Learning & Multi-Modal Learning",
                duration: 20,
                notes: "Shared representations. CLIP, DALL-E architecture. Vision-language models."
            }
        ]
    },
    {
        day: 34,
        category: "Causal Inference & Experimentation",
        tasks: [
            {
                title: "Causal Inference in ML",
                duration: 60,
                notes: "Correlation vs causation. Causal graphs, do-calculus. Counterfactual reasoning."
            },
            {
                title: "Experimental Design for ML",
                duration: 40,
                notes: "A/B testing pitfalls. Novelty effects. Network effects. Your KPI automation work."
            },
            {
                title: "Treatment Effect Estimation",
                duration: 20,
                notes: "Propensity score matching. Instrumental variables."
            }
        ]
    },
    {
        day: 35,
        category: "Behavioral - Googliness",
        tasks: [
            {
                title: "Google Leadership Principles for ML Roles",
                duration: 60,
                notes: "Innovation, user focus, collaboration. Prepare 10 STAR stories covering all areas."
            },
            {
                title: "Ethical AI & Responsible ML Practice",
                duration: 40,
                notes: "Bias in ML, fairness metrics. Your AI governance work at Rivian. Google's AI ethics."
            },
            {
                title: "Practice: 'Why Google?' & 'Why ML?'",
                duration: 20,
                notes: "Authentic stories. Connect to your research and Vertex AI experience."
            }
        ]
    },

    // Week 8: Final Polish & Interview Readiness
    {
        day: 36,
        category: "ML Coding Interview Prep",
        tasks: [
            {
                title: "Implement Backpropagation from Scratch",
                duration: 40,
                notes: "Chain rule application. Computational graph. Manual gradient calculation."
            },
            {
                title: "Implement Attention Mechanism",
                duration: 40,
                notes: "Scaled dot-product attention. Multi-head attention. NumPy or PyTorch."
            },
            {
                title: "Implement Mini-batch SGD with Momentum",
                duration: 40,
                notes: "Vectorization. Learning rate scheduling. Convergence behavior."
            }
        ]
    },
    {
        day: 37,
        category: "ML Theory Interview",
        tasks: [
            {
                title: "ML Theory Questions Practice (50 questions)",
                duration: 70,
                notes: "Why does batch norm work? Explain dropout. What is gradient clipping?"
            },
            {
                title: "Math for ML Interview Questions",
                duration: 50,
                notes: "Matrix calculus, probability distributions, optimization theory."
            }
        ]
    },
    {
        day: 38,
        category: "ML System Design - Mock",
        tasks: [
            {
                title: "Design Spam Classification System",
                duration: 60,
                notes: "Gmail spam filter. Online learning. Adversarial users. Precision vs recall."
            },
            {
                title: "Design Voice Assistant (Google Assistant)",
                duration: 60,
                notes: "Speech recognition, NLU, dialogue management, TTS. Low latency. Multi-turn."
            }
        ]
    },
    {
        day: 39,
        category: "Data Structures & Algorithms for ML Engineers",
        tasks: [
            {
                title: "Essential DS&A for ML: Arrays, Hash Tables, Trees",
                duration: 60,
                notes: "LeetCode easy/medium problems. Focus on ML-relevant patterns."
            },
            {
                title: "Graph Algorithms for ML",
                duration: 40,
                notes: "BFS, DFS, shortest path. Graph representation. Your GNN knowledge."
            },
            {
                title: "Dynamic Programming for ML",
                duration: 20,
                notes: "Sequence alignment, edit distance (NLP applications)."
            }
        ]
    },
    {
        day: 40,
        category: "Full Mock Interview Day",
        tasks: [
            {
                title: "Complete Mock Interview Loop (4 rounds)",
                duration: 90,
                notes: "ML coding, ML system design, ML theory, behavioral. Simulate real Google interview."
            },
            {
                title: "Review & Identify Gaps",
                duration: 30,
                notes: "What went well? What needs improvement? Final adjustments for last 20 days."
            }
        ]
    },

    // Week 9-10: Advanced Projects & Case Studies
    {
        day: 41,
        category: "MLOps Projects & Monitoring",
        tasks: [
            {
                title: "Build End-to-End ML Project: Image Classification",
                duration: 50,
                notes: "Data collection, preprocessing, model training, deployment. Document everything."
            },
            {
                title: "Add MLOps: CI/CD, Monitoring, Versioning",
                duration: 40,
                notes: "GitHub Actions, MLflow, model registry. Production-ready pipeline."
            },
            {
                title: "🔍 Implement Model Monitoring & Drift Detection",
                duration: 30,
                notes: "Evidently AI or Whylabs. Monitor data drift, concept drift, model performance degradation. Set up alerts."
            }
        ]
    },
    {
        day: 42,
        category: "ML Projects Deep Dive",
        tasks: [
            {
                title: "Build NLP Project: Sentiment Analysis API",
                duration: 70,
                notes: "Fine-tune BERT. Deploy with FastAPI. Docker container. Your REST API experience."
            },
            {
                title: "Add Production Features: Caching, Rate Limiting",
                duration: 50,
                notes: "Redis caching. Authentication. Logging. Your Django/Flask background."
            }
        ]
    },
    {
        day: 43,
        category: "ML Case Studies",
        tasks: [
            {
                title: "Analyze Google Research Papers (3 papers)",
                duration: 70,
                notes: "Recent NeurIPS, ICML papers from Google. Understand problem, approach, results."
            },
            {
                title: "Practice Explaining Research Papers",
                duration: 50,
                notes: "Pretend you're presenting to non-technical stakeholders. Simplify complexity."
            }
        ]
    },
    {
        day: 44,
        category: "ML System Design - Real Cases",
        tasks: [
            {
                title: "Case Study: Netflix Recommendation System",
                duration: 60,
                notes: "Analyze Netflix's actual approach. Matrix factorization, deep learning, bandits."
            },
            {
                title: "Case Study: Tesla Autopilot",
                duration: 60,
                notes: "Computer vision, sensor fusion, path planning. Your ADAS experience relevance."
            }
        ]
    },
    {
        day: 45,
        category: "Mock Interview - Intensive",
        tasks: [
            {
                title: "Back-to-Back Mock Interviews (3 rounds)",
                duration: 90,
                notes: "No breaks. Simulate real interview fatigue. ML coding, system design, theory."
            },
            {
                title: "Peer Review Session",
                duration: 30,
                notes: "Get feedback from experienced ML engineers. Refine communication."
            }
        ]
    },

    // Week 11: Specialization & Advanced Topics
    {
        day: 46,
        category: "Advanced NLP",
        tasks: [
            {
                title: "Prompt Engineering for LLMs",
                duration: 60,
                notes: "Chain-of-thought, few-shot prompting. Your Beacon AI prompt orchestration work."
            },
            {
                title: "LLM Fine-tuning: LoRA, QLoRA, PEFT",
                duration: 40,
                notes: "Parameter-efficient fine-tuning. Adapter layers. Memory-efficient training."
            },
            {
                title: "Code: Fine-tune Llama or GPT on Custom Data",
                duration: 20,
                notes: "HuggingFace Transformers. Supervised fine-tuning vs RLHF."
            }
        ]
    },
    {
        day: 47,
        category: "Advanced Computer Vision",
        tasks: [
            {
                title: "Vision Transformers (ViT) Deep Dive",
                duration: 60,
                notes: "Patch embeddings, self-attention for images. ViT vs CNNs trade-offs."
            },
            {
                title: "Multi-Modal Models: CLIP, Flamingo",
                duration: 40,
                notes: "Vision-language pretraining. Zero-shot classification. Your CV+NLP potential."
            },
            {
                title: "3D Vision & Point Clouds",
                duration: 20,
                notes: "PointNet, LiDAR processing. Your 2D-to-3D mapping experience at Continental."
            }
        ]
    },
    {
        day: 48,
        category: "ML at Scale",
        tasks: [
            {
                title: "Federated Learning",
                duration: 50,
                notes: "Privacy-preserving ML. Model aggregation. Google's keyboard prediction use case."
            },
            {
                title: "Online Learning & Streaming ML",
                duration: 50,
                notes: "Incremental updates. Concept drift handling. Real-time model updates."
            },
            {
                title: "Neural Architecture Search (NAS)",
                duration: 20,
                notes: "AutoML. EfficientNet design. Computational cost vs performance."
            }
        ]
    },
    {
        day: 49,
        category: "ML Infrastructure & Tools",
        tasks: [
            {
                title: "Kubernetes for ML Workloads",
                duration: 60,
                notes: "KubeFlow, Seldon. GPU scheduling. Your Docker/K8s experience at Bosch."
            },
            {
                title: "ML Model Serving: TensorFlow Serving, Triton",
                duration: 40,
                notes: "Batching, model ensembles. Latency optimization. Your microservices background."
            },
            {
                title: "Review: Your End-to-End ML Experience",
                duration: 20,
                notes: "Rivian Databricks pipelines, Bosch cloud-native ML services, Continental ADAS."
            }
        ]
    },
    {
        day: 50,
        category: "Mock Interview - Final Round",
        tasks: [
            {
                title: "Intensive Mock Interview Day (Full Loop)",
                duration: 100,
                notes: "4-5 rounds: ML coding, ML design, ML theory, behavioral, hiring manager. Treat as real."
            },
            {
                title: "Final Feedback & Adjustments",
                duration: 20,
                notes: "Last-minute improvements. Communication clarity. Confidence building."
            }
        ]
    },

    // Week 12: Final Preparation & Confidence Building
    {
        day: 51,
        category: "Resume & Portfolio Polish",
        tasks: [
            {
                title: "Update Resume: Quantify All Achievements",
                duration: 60,
                notes: "ML metrics: accuracy, latency, cost savings. 30% false positive reduction in Beacon."
            },
            {
                title: "GitHub Portfolio: Pin Best ML Projects",
                duration: 40,
                notes: "Clean READMEs. Add demo videos. Deployment links. Showcase end-to-end skills."
            },
            {
                title: "LinkedIn: Optimize for ML Roles",
                duration: 20,
                notes: "Add skills: PyTorch, Vertex AI, MLOps. Recommendations from colleagues."
            }
        ]
    },
    {
        day: 52,
        category: "ML Research Awareness",
        tasks: [
            {
                title: "Read Latest Google AI Blog Posts (10 posts)",
                duration: 60,
                notes: "Understand Google's current ML focus areas. Gemini, PaLM, Bard updates."
            },
            {
                title: "Follow Google AI Researchers on Twitter/LinkedIn",
                duration: 30,
                notes: "Jeff Dean, Demis Hassabis. Stay updated on Google Brain/DeepMind."
            },
            {
                title: "Prepare Questions About Google's ML Direction",
                duration: 30,
                notes: "Thoughtful questions for interviewers. Show genuine interest."
            }
        ]
    },
    {
        day: 53,
        category: "Behavioral Interview Polish",
        tasks: [
            {
                title: "Refine All STAR Stories (15 stories)",
                duration: 60,
                notes: "Beacon AI, Databricks pipelines, ADAS, computational trust, automation tools."
            },
            {
                title: "Practice Leadership & Conflict Questions",
                duration: 40,
                notes: "Product owner role, Scrum master, cross-functional collaboration."
            },
            {
                title: "Record Yourself: Top 10 Behavioral Questions",
                duration: 20,
                notes: "Watch for filler words, clarity, confidence. 2-3 min per story max."
            }
        ]
    },
    {
        day: 54,
        category: "ML Coding Speed Practice",
        tasks: [
            {
                title: "Timed ML Coding Challenges (5 problems)",
                duration: 70,
                notes: "30 min per problem. Implement from scratch: KNN, Decision Tree, Linear Reg, etc."
            },
            {
                title: "PyTorch/TensorFlow Speed Coding",
                duration: 50,
                notes: "Build simple CNN, RNN, Transformer in under 20 min each. Muscle memory."
            }
        ]
    },
    {
        day: 55,
        category: "ML System Design Speed Practice",
        tasks: [
            {
                title: "Rapid Design Practice: 5 Systems in 5 Hours",
                duration: 60,
                notes: "Recommendation, search, fraud detection, ad prediction, chatbot. 1 hr each."
            },
            {
                title: "Framework Practice: Consistent Approach",
                duration: 40,
                notes: "Requirements → High-level design → ML formulation → Training/Serving → Monitoring."
            },
            {
                title: "Common Pitfalls Review",
                duration: 20,
                notes: "Forgetting monitoring, ignoring latency, not discussing tradeoffs."
            }
        ]
    },

    // Days 56-58: Light Review & Mental Preparation
    {
        day: 56,
        category: "Light Review - ML Theory",
        tasks: [
            {
                title: "Review ML Flashcards (100 cards)",
                duration: 60,
                notes: "Key concepts, formulas, architectures. Active recall practice."
            },
            {
                title: "Watch ML Interview Prep Videos",
                duration: 40,
                notes: "YouTube: ML system design examples. Common mistakes to avoid."
            },
            {
                title: "Relax: Short Walk or Meditation",
                duration: 20,
                notes: "Don't over-stress. Trust your preparation."
            }
        ]
    },
    {
        day: 57,
        category: "Light Review - Your Experience",
        tasks: [
            {
                title: "Review Your Resume Line-by-Line",
                duration: 50,
                notes: "Be ready to explain every project, technology, metric. Deep knowledge."
            },
            {
                title: "Practice: 'Walk me through your resume'",
                duration: 40,
                notes: "5-min version. Highlight ML projects, impact, learnings."
            },
            {
                title: "Prepare Setup: Laptop, Internet, Backup Plan",
                duration: 30,
                notes: "Test video, audio. Coderpad practice. Whiteboard/paper ready."
            }
        ]
    },
    {
        day: 58,
        category: "Mental Preparation",
        tasks: [
            {
                title: "Visualization: Successful Interview",
                duration: 40,
                notes: "Imagine yourself calm, articulate, solving problems confidently."
            },
            {
                title: "Review Your 'Why Google' & 'Why ML' Stories",
                duration: 40,
                notes: "Authentic passion. Your research, Vertex AI experience, Rivian impact."
            },
            {
                title: "Confidence Building: List Your Strengths",
                duration: 40,
                notes: "PhD-level research, 6+ YoE, production ML, MLOps, CV, NLP, RL. You're ready!"
            }
        ]
    },

    // Day 59: Pre-Interview Day
    {
        day: 59,
        category: "Final Day Before Interview",
        tasks: [
            {
                title: "Very Light Review: Skim Notes Only",
                duration: 40,
                notes: "No new learning. Just refresh key concepts. Trust your preparation."
            },
            {
                title: "Self-Care: Exercise, Healthy Meal, Good Sleep",
                duration: 50,
                notes: "Physical readiness = mental sharpness. 8 hours sleep minimum."
            },
            {
                title: "Prepare Interview Day Logistics",
                duration: 30,
                notes: "Outfit, snacks, water, backup internet. Set alarms. Relax tonight."
            }
        ]
    },

    // Day 60: Interview Day
    {
        day: 60,
        category: "Interview Day",
        tasks: [
            {
                title: "Morning Prep: Review Your 'Cheat Sheet'",
                duration: 30,
                notes: "One-pagers: ML formulas, system design framework, STAR stories."
            },
            {
                title: "Calm Your Nerves: Breathing Exercises",
                duration: 20,
                notes: "Box breathing: 4-4-4-4. You've prepared extensively. You've got this!"
            },
            {
                title: "Post-Interview: Reflect & Send Thank You Notes",
                duration: 30,
                notes: "What went well? What could improve? Thank interviewers. Celebrate effort! 🎉"
            }
        ]
    }
];

// Daily Coding Practice Problems (LeetCode-style)
const codingPractice = [
    // Week 1
    { day: 1, problems: ["Two Sum (Easy)", "Valid Parentheses (Easy)", "Reverse Linked List (Easy)"], category: "Arrays & Strings" },
    { day: 2, problems: ["Maximum Subarray (Medium)", "Container With Most Water (Medium)", "3Sum (Medium)"], category: "Arrays & Sliding Window" },
    { day: 3, problems: ["Longest Substring Without Repeating (Medium)", "Group Anagrams (Medium)", "Top K Frequent Elements (Medium)"], category: "Hash Tables" },
    { day: 4, problems: ["Implement strStr() (Easy)", "Longest Common Prefix (Easy)", "Valid Palindrome (Easy)"], category: "String Manipulation" },
    { day: 5, problems: ["Merge Two Sorted Lists (Easy)", "Remove Nth Node (Medium)", "Add Two Numbers (Medium)"], category: "Linked Lists" },
    
    // Week 2
    { day: 6, problems: ["Binary Tree Inorder Traversal (Easy)", "Maximum Depth (Easy)", "Same Tree (Easy)"], category: "Binary Trees" },
    { day: 7, problems: ["Validate BST (Medium)", "Binary Tree Level Order (Medium)", "Construct Tree from Preorder/Inorder (Medium)"], category: "BST & Tree Traversal" },
    { day: 8, problems: ["Lowest Common Ancestor (Medium)", "Serialize/Deserialize Tree (Hard)", "Binary Tree Maximum Path Sum (Hard)"], category: "Advanced Trees" },
    { day: 9, problems: ["Number of Islands (Medium)", "Clone Graph (Medium)", "Course Schedule (Medium)"], category: "Graph BFS/DFS" },
    { day: 10, problems: ["Word Ladder (Hard)", "Network Delay Time (Medium)", "Cheapest Flights (Medium)"], category: "Graph Algorithms" },
    
    // Week 3-4
    { day: 11, problems: ["Climbing Stairs (Easy)", "House Robber (Easy)", "Coin Change (Medium)"], category: "Dynamic Programming Intro" },
    { day: 12, problems: ["Longest Increasing Subsequence (Medium)", "Word Break (Medium)", "Partition Equal Subset (Medium)"], category: "DP - Subsequences" },
    { day: 13, problems: ["Edit Distance (Hard)", "Longest Common Subsequence (Medium)", "Distinct Subsequences (Hard)"], category: "DP - Strings" },
    { day: 14, problems: ["Unique Paths (Medium)", "Minimum Path Sum (Medium)", "Dungeon Game (Hard)"], category: "DP - Grid" },
    { day: 15, problems: ["Permutations (Medium)", "Combinations (Medium)", "Subsets (Medium)"], category: "Backtracking" },
    
    // Continue through all 60 days
    { day: 20, problems: ["Implement Trie (Medium)", "Word Search II (Hard)", "Design Add Search Words (Medium)"], category: "Trie" },
    { day: 25, problems: ["Kth Largest Element (Medium)", "Merge K Sorted Lists (Hard)", "Find Median from Data Stream (Hard)"], category: "Heap/Priority Queue" },
    { day: 30, problems: ["Sliding Window Maximum (Hard)", "Trapping Rain Water (Hard)", "Largest Rectangle in Histogram (Hard)"], category: "Stack/Queue" },
    { day: 40, problems: ["Median of Two Sorted Arrays (Hard)", "Kth Smallest in BST (Medium)", "Search in Rotated Array (Medium)"], category: "Binary Search" },
    { day: 50, problems: ["Regular Expression Matching (Hard)", "Wildcard Matching (Hard)", "Interleaving String (Hard)"], category: "Advanced DP" },
    { day: 60, problems: ["Review your top 10 hardest problems"], category: "Final Review" }
];

// System Design Practice Problems
const systemDesignProblems = [
    { day: 7, problem: "Design URL Shortener (bit.ly)", topics: ["Hashing", "Database sharding", "Rate limiting", "Caching"], difficulty: "Medium" },
    { day: 11, problem: "Design Instagram/Twitter Feed", topics: ["Fan-out on write/read", "Timeline generation", "Caching", "CDN"], difficulty: "Medium" },
    { day: 14, problem: "Design Netflix Recommendation System", topics: ["Collaborative filtering", "Matrix factorization", "Real-time updates", "A/B testing"], difficulty: "Hard" },
    { day: 17, problem: "Design MLOps Platform (Feature Store + Model Registry)", topics: ["Feature engineering pipeline", "Model versioning", "Deployment automation", "Monitoring"], difficulty: "Hard" },
    { day: 19, problem: "Design Google Search (High-Level)", topics: ["Crawling", "Indexing", "Ranking (PageRank)", "Distributed storage"], difficulty: "Hard" },
    { day: 21, problem: "Design YouTube Video Streaming", topics: ["CDN", "Video encoding", "Chunked upload", "View count aggregation"], difficulty: "Hard" },
    { day: 24, problem: "Design Fraud Detection System", topics: ["Real-time ML scoring", "Feature engineering", "Rule engine + ML", "Alert system"], difficulty: "Hard" },
    { day: 28, problem: "Design Uber/Lyft Ride Matching", topics: ["Geohashing", "Real-time matching", "ETA prediction", "Surge pricing"], difficulty: "Hard" },
    { day: 30, problem: "Design WhatsApp/Messenger", topics: ["WebSockets", "Message queue", "End-to-end encryption", "Delivery receipts"], difficulty: "Medium" },
    { day: 32, problem: "Design Google Photos Search", topics: ["Image embeddings", "Vector similarity search", "Face clustering", "Privacy"], difficulty: "Hard" },
    { day: 35, problem: "Design Ad Click Prediction System", topics: ["Feature store", "Online learning", "CTR modeling", "Real-time bidding"], difficulty: "Hard" },
    { day: 38, problem: "Design Spam Classification (Gmail)", topics: ["Online learning", "Feature extraction", "Adversarial users", "Precision/Recall"], difficulty: "Medium" },
    { day: 40, problem: "Design Voice Assistant (Google Assistant)", topics: ["Speech-to-text", "NLU", "Dialogue management", "Text-to-speech", "Low latency"], difficulty: "Hard" },
    { day: 42, problem: "Design MLOps CI/CD Pipeline", topics: ["Automated testing", "Model validation", "Canary deployment", "Rollback strategies", "Blue-green deployment"], difficulty: "Hard" },
    { day: 44, problem: "Design Autonomous Driving Perception", topics: ["Sensor fusion", "Real-time CV", "Object tracking", "Path planning", "Safety"], difficulty: "Expert" },
    { day: 48, problem: "Design Real-Time ML Model Serving at Scale", topics: ["Model versioning", "A/B testing", "Canary deployment", "Monitoring", "Auto-scaling"], difficulty: "Hard" },
    { day: 52, problem: "Design News Feed Ranking Algorithm", topics: ["ML ranking", "Personalization", "Content moderation", "Diversity"], difficulty: "Hard" },
    { day: 56, problem: "Design Distributed ML Training System", topics: ["Parameter server", "Data parallelism", "Fault tolerance", "GPU scheduling"], difficulty: "Expert" }
];

// ML Coding Challenges (Implement from Scratch)
const mlCodingChallenges = [
    { day: 1, challenge: "Linear Regression with Gradient Descent", implement: "Fit, predict, cost function", language: "NumPy only" },
    { day: 3, challenge: "Logistic Regression", implement: "Sigmoid, cost function, gradient descent", language: "NumPy only" },
    { day: 5, challenge: "K-Nearest Neighbors", implement: "Distance metrics, majority voting", language: "NumPy only" },
    { day: 8, challenge: "Decision Tree Classifier", implement: "Gini impurity, recursive splitting", language: "Python" },
    { day: 12, challenge: "K-Means Clustering", implement: "Lloyd's algorithm, convergence check", language: "NumPy" },
    { day: 15, challenge: "Neural Network (2-layer)", implement: "Forward pass, backprop, train", language: "NumPy only" },
    { day: 18, challenge: "Convolutional Layer", implement: "Conv2D forward pass", language: "NumPy" },
    { day: 22, challenge: "Attention Mechanism", implement: "Scaled dot-product attention", language: "NumPy or PyTorch" },
    { day: 25, challenge: "Batch Normalization", implement: "Forward and backward pass", language: "NumPy" },
    { day: 30, challenge: "Adam Optimizer", implement: "Momentum + RMSProp", language: "NumPy" },
    { day: 36, challenge: "Backpropagation from Scratch", implement: "Chain rule, computational graph", language: "NumPy" },
    { day: 40, challenge: "Mini-batch SGD with Momentum", implement: "Batch processing, momentum update", language: "NumPy" },
    { day: 45, challenge: "Simple RNN Cell", implement: "Forward pass, BPTT", language: "NumPy" },
    { day: 50, challenge: "Dropout Layer", implement: "Training vs inference mode", language: "NumPy" }
];

// Learning topics organized by category
const learningTopics = {
    "Machine Learning Fundamentals": [
        "Supervised Learning (Linear/Logistic Regression, Trees, Ensembles)",
        "Unsupervised Learning (Clustering, PCA, Anomaly Detection)",
        "Bias-Variance Tradeoff & Regularization",
        "Cross-validation & Model Evaluation",
        "Feature Engineering & Selection"
    ],
    "Deep Learning": [
        "Neural Networks (Forward/Backpropagation, Activation Functions)",
        "CNNs (ResNet, VGG, Inception, Transfer Learning)",
        "RNNs & LSTMs (Sequence Modeling, Attention)",
        "Transformers (BERT, GPT, T5, ViT)",
        "Optimization (SGD, Adam, Learning Rate Scheduling)"
    ],
    "Computer Vision": [
        "Object Detection (YOLO, R-CNN, SSD)",
        "Image Segmentation (U-Net, Mask R-CNN)",
        "2D-to-3D Mapping & Depth Estimation",
        "Vision Transformers",
        "Multi-Modal Models (CLIP, DALL-E)"
    ],
    "Natural Language Processing": [
        "Transformers & Attention Mechanisms",
        "Large Language Models (GPT, BERT, T5)",
        "Prompt Engineering & Few-Shot Learning",
        "Retrieval-Augmented Generation (RAG)",
        "Fine-tuning (LoRA, PEFT)"
    ],
    "Reinforcement Learning": [
        "MDP & Bellman Equations",
        "Q-Learning & DQN",
        "Policy Gradients (REINFORCE, PPO, A3C)",
        "Actor-Critic Methods",
        "Applications in Robotics & Autonomous Systems"
    ],
    "MLOps & Production ML": [
        "ML Pipelines (Training, Serving, Monitoring)",
        "CI/CD for ML (Jenkins, GitLab CI, GitHub Actions)",
        "Feature Stores & Data Versioning (Feast, Tecton)",
        "Model Deployment (TensorFlow Serving, TorchServe, SageMaker)",
        "Model Monitoring & Drift Detection (Evidently, Whylabs)",
        "A/B Testing & Experimentation Platforms",
        "Model Registry & Version Control (MLflow, Weights & Biases)",
        "Distributed Training (Horovod, PyTorch DDP, Ray)",
        "Infrastructure as Code (Terraform, CloudFormation)",
        "Kubernetes for ML (KubeFlow, Seldon Core)",
        "Observability (Prometheus, Grafana, CloudWatch)",
        "Data Quality & Validation (Great Expectations, Deequ)"
    ],
    "ML System Design": [
        "Recommendation Systems (Collaborative Filtering, Content-Based)",
        "Search Ranking (Learning to Rank, NDCG)",
        "Fraud Detection & Anomaly Detection",
        "Ad Click Prediction (CTR)",
        "Real-time Personalization"
    ],
    "Advanced Topics": [
        "Graph Neural Networks (GCN, GAT, GraphSAGE)",
        "Generative AI (GANs, VAEs, Diffusion Models)",
        "Federated Learning",
        "Neural Architecture Search (AutoML)",
        "Causal Inference & Experimentation"
    ],
    "ML Math & Theory": [
        "Linear Algebra (Matrix Operations, SVD, Eigendecomposition)",
        "Probability & Statistics (Bayes Theorem, Distributions)",
        "Optimization Theory (Gradient Descent Variants)",
        "Information Theory (Entropy, KL Divergence)",
        "Calculus for ML (Chain Rule, Backpropagation Math)"
    ],
    "Tools & Frameworks": [
        "PyTorch & TensorFlow/JAX",
        "Scikit-learn & XGBoost",
        "Hugging Face Transformers",
        "MLflow & Kubeflow",
        "Databricks & Vertex AI"
    ],
    "Data Engineering for ML": [
        "Feature Engineering & ETL Pipelines",
        "Data Quality & Validation",
        "Spark for ML (MLlib, PySpark)",
        "Snowflake & Data Warehousing",
        "Vector Databases (Pinecone, ChromaDB)"
    ],
    "ML Security & Ethics": [
        "Adversarial Machine Learning",
        "Model Robustness & Uncertainty",
        "Fairness & Bias in ML",
        "Explainable AI (SHAP, LIME)",
        "AI Governance & Responsible AI"
    ]
};

// Export for use in app.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { preparationPlan, learningTopics, codingPractice, systemDesignProblems, mlCodingChallenges };
}
