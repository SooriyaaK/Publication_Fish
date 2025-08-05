
Neighbors' Influence on Predicting Focal Fish's Velocity

Overview:
In order to find answers for our research question the respitory was created.

RQ1: How well can we predict fish behavior based on its neighbors' cues?

RQ2: Which neighbors provide the most accurate prediction of the fish's velocity?

RQ3: What kind of information regarding the neighbor information is the basis for this choice?

To answer the research questions:
- developed an evolutionary algorithm to optimize prediction accuracy.
- visualized weight distributions to interpret each group member's influence.
- simulated behavior cues (bearing, distance, orientation)
- tested if the wall has a signifiant impact on predicting the focal fish's velocity.
- compared predicted and actual agent movements to evaluate model performance.

Important functionalities:
- Loss Evaluation: 
    -> Compute and save various loss metrics (cosine similarity, MSE on kick times and trajectories) 
       across different bias configurations (Bias Strategy) and Sorting Criteria(Bearing, Distance, Orientation).

- Weight Visualization: 
    -> Plot normalized weight vectors per EA run, highlighting the best-performing run.

Trajectory Analysis: 
-> Load actual and predicted positions/velocities and generate trajectory and quiver plots.

Each loss function consists of their Sorting Criteria(Bearing, Distance, Orientation)

Cosine - folder:
    Cosine/Bearing 
    Cosine/Distance 
    Cosine/Orientation


mse_loss_kicktime - folder:
    mse_loss_kicktime/Bearing 
    mse_loss_kicktime/Distance 
    mse_loss_kicktime/Orientation

mse_loss_trajectory - folder:
    mse_loss_trajectory/Bearing 
    mse_loss_trajectory/Distance 
    mse_loss_trajectory/Orientation


Each of the Sorting Criteria folder consists of: Agent 3
    mse_loss_cosine/Bearing/Agent_3
    mse_loss_kicktime/Bearing/Agent_3
    mse_loss_trajectory/Bearing/Agent_3


Agent 3 consists of these four files:  Bias_Wall_Hyperparameter.ipynb, step1.ipynb, step2.ipynb, step4.ipynb

mse_loss_trajectory/Bearing/Agent_3/Bias_Wall_Hyperparameter.ipynb 
mse_loss_trajectory/Bearing/Agent_3/step1.ipynb 
mse_loss_trajectory/Bearing/Agent_3/step2.ipynb 
mse_loss_trajectory/Bearing/Agent_3/step4.ipynb
   
step1.ipynb: Preprocessing data.
step2.ipynb: Feature engineering and velocity calculations.
step4.ipynb: Wall Behavior.
Bias_Wall_Hyperparameter.ipynb: EA - crossover: 0.6, mutation_rate: 0.2. k = 8

LossCompare folder consists of:
    LossCompare/visualization.ipynb 
    LossCompare/statistic.ipynb

visualization.ipynb:
-Plot comparing weights for each loss functions
-Plot comparing predicted vs actual
-PCA
-Plot Experiment 151
-Boxplot comparing the mean 

statistic.ipynb: 
- Shapiro Wilk Test,
- Levenne Test,
- ANOVA
- Tukey's Honest Significant Difference (HSD) test,
- T test 

Libraries:
-Python (3.12.3)
-NumPy
-Pandas
-Matplotlib
-SciPy
-Scikit-learn
-Seaborn