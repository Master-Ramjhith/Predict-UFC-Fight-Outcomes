Machine Learning Mini-Project: UFC Fight Outcome Prediction
1. Problem Statement
The objective of this project was to develop a high-accuracy, robust machine learning model capable of predicting the winner of a professional UFC (Ultimate Fighting Championship) fight based exclusively on pre-fight fighter statistics and historical data. This serves as a significant challenge due to the high variance, limited data points per fighter, and complex psychological factors inherent in combat sports.
2. Approach & Methodology
Data Preparation and Engineering
The raw dataset, spanning 1994–2025, contained career statistics (r_splm, b_str_acc), physical attributes, and fight metadata. The core challenge was transforming this static, cumulative data into a dynamic feature that captures a fighter's current form.
1.	EWMA Feature Engineering: We utilized Exponentially Weighted Moving Averages (EWMA) on key in-fight performance metrics (Knockdowns, Significant Strikes Landed, Takedowns, and Control Time). EWMA applies exponential decay, giving significantly more weight to a fighter's most recent performances. The difference between the Red fighter's EWMA and the Blue fighter's EWMA was used as a feature, resulting in highly predictive, time-sensitive variables.
2.	Relative Features: All static pre-fight career metrics (e.g., career Striking Accuracy, Takedown Defense, Height, Weight) were transformed into difference features (Red fighter stat – Blue fighter stat).
3.	Target Variable: The target was binary: 1 (Red Fighter Win), 0 (Blue Fighter Win).
Model Selection and Training
The XGBoost Classifier was selected for its performance, speed, and ability to handle complex non-linear relationships, which is ideal for a high-dimensional feature set derived from time-series data.
Optimization: To ensure rapid execution and maximum accuracy, the hyperparameter tuning phase was skipped. The model was trained using a set of pre-optimized parameters (n_estimators=800, learning_rate=0.03, max_depth=5) known to perform well on large, tree-based datasets.
3. Implementation Overview & Results
The final model was trained on 19 generated features using a 80/20 train-test split. The pipeline includes data loading, cleaning (removing non-binary outcomes), EWMA calculation, feature scaling (StandardScaler), model training, and evaluation.
Metric	Value	Model	Execution Time	Total Features
Final Accuracy (Test Set)	0.6880	XGBoost Classifier	29.73 seconds	19
ROC AUC Score	0.7452	-	-	-
The key takeaway from the Feature Importance Plot was that the newly engineered EWMA difference features (especially diff_ewma_td_landed and diff_ewma_sig_str_landed) were highly ranked, validating the hypothesis that recent performance form is more predictive than static career averages.
4. Conclusions & Challenges
Conclusion: The model achieved a robust accuracy of 68.8% and an excellent AUC score of 0.7452, demonstrating strong predictive power significantly above random chance. The total execution time of 29.73 seconds ensured the solution was both high-performing and highly efficient.
Challenges: The primary challenge was addressing look-ahead bias during EWMA calculation, which was mitigated by ensuring the EWMA calculation for any given fight only included data from previous fights (using the shift(1) function). Additionally, handling missing or inconsistent ID values within the heterogeneous dataset required robust error checking and data type casting.

