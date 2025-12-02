A Hybrid Quantum Machine Learning Algorithm to predict and classify risk of Coronary Artery Disease (CAD) from a dataset of Photoplethysmography (PPG) Signals. This project serves as a proof of concept for the potential of Qunatum Computing and Hybrid Quantum Computing in real-time prediction scenarios, which can greatly benifit from the vastly faster computing speed of Quantum Computers.

This model compares the performance of ADASYN-tuned Linear Regression Model and a 4-Qubit Circuit with two superposed gates, in the prediction and classification of CAD risk in a PPG signal based on a probability score, dervied from analyzing 16 Key features extracted from the base data.


**Dataset:**

1,250 clinically annotated, Multi-parameter ICU data segments from PhysioNet 2015 Computing in Cardiology Challenge. Each data segment was 5 minutes long (for real time analysis), ending at the time of the alarm. For retrospective analysis, we provided a further 30 seconds of data after the alarm was triggered. The dataset contained multiple cardiograms (ECGs. EEGs, PPGs, etc.), and so PPG signals were extracted from the raw data using keywords such as **pleth**, **ppg** or **photopleth**.

<img width="1100" height="1000" alt="Visualize_PPG" src="https://github.com/user-attachments/assets/125ed1ed-36d5-4feb-84cf-2010c6fb36ff" />

Out of the 41 time and frequency domain features, 16 key features were chosen based on feature importance that were used in the training of the classical machine learning algorithm and in amplitude encoding in the 4-Qubit Quantum Circuit. 

<img width="1000" height="1000" alt="feature_importance_top20" src="https://github.com/user-attachments/assets/03355a24-0d38-46c5-81d1-54d6f4f5229e"/>


**Classical Machine Learning:**

scikit-learn python library was used to access **SVM RBF**, **SVM Linear**, **Random Forest**, **Neural Network** and **Logistic Regression**, out of which **Logistic Regression** achieved the highest F1 score.

<img width="1100" height="1000" alt="all_confusion_matrices" src="https://github.com/user-attachments/assets/baca48f0-082f-48e1-ae42-74874891b5a4" />

Next, multiple optimization algorithms such as **SMOTE**, **Borderline SMOTE**, and **ADASYN** were used to optimize the F1 score of Linear Regression by fine tuning parameters and creating synthetic samples in the minority class. 

<img width="1100" height="1000" alt="metrics_comparison_all" src="https://github.com/user-attachments/assets/c240dda2-9bfe-4c04-8493-7961eeeaf131" />


**Quantum Machine Learning:**

Qiskit Quantum emulator library to encode the key features in a 4-Qubit Quantum circuit with ampltude encoding and two entanglement gates that can rotate to a superposition of all the key features to get the most optimal result. 

<img width="1100" height="1000" alt="quantum_training_history" src="https://github.com/user-attachments/assets/928fcf61-6978-4c1a-bd23-fb093e1d8fd1" />


**Final Results:**

<img width="5370" height="5895" alt="CAD_Flagged_Signals_Grid_2025-10-30_22-07-22" src="https://github.com/user-attachments/assets/c2d1c0c0-77cf-4cf7-a573-ce9dd1e288f9" />


