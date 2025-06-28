# Credit Risk Assessment Product

This repository contains the codebase for developing a credit risk assessment product. The goal is to categorize users as high or low risk, assign credit scores, and predict optimal loan amounts and durations based on transactional data.

## Credit Scoring Business Understanding

Credit risk is the possibility of a financial loss due to a borrower's failure to repay a loan or meet contractual obligations. Managing credit risk is paramount for financial institutions to maintain solvency and profitability.

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord, and its successors (Basel III), significantly influence how banks manage and quantify risk. Its core pillars are:
1.  **Minimum Capital Requirements (Pillar 1):** Mandates that banks hold sufficient capital to cover credit, operational, and market risks. It provides methodologies (Standardized, Internal Rating-Based - IRB) for calculating Risk-Weighted Assets (RWA). For IRB approaches, banks develop their own internal models.
2.  **Supervisory Review Process (Pillar 2):** Encourages regulators to review banks' risk management frameworks and capital adequacy.
3.  **Market Discipline (Pillar 3):** Requires banks to disclose risk exposures, assessment processes, and capital adequacy to foster market transparency.

This emphasis on rigorous risk measurement directly influences our need for an interpretable and well-documented model in several ways:

* **Regulatory Compliance:** Financial institutions operating under Basel II/III regulations must demonstrate to supervisors that their models are sound, robust, and comply with strict guidelines. An interpretable model facilitates this demonstration by allowing regulators to understand the rationale behind risk assignments.
* **Validation and Auditability:** Models used for capital calculation or credit decisions are subject to independent validation and regular audits. Clear documentation of features, model logic, and assumptions, along with interpretability, makes this validation process efficient and reliable.
* **Risk Management Decisions:** Beyond compliance, bank management needs to understand *why* a customer is categorized as high or low risk to make informed decisions about lending policies, interest rates, and loan structures. An interpretable model provides actionable insights for risk mitigation strategies.
* **Challenging Model Outputs:** If a model produces unexpected or seemingly erroneous results, interpretability allows analysts to pinpoint the specific factors driving the outcome, diagnose issues, and improve the model or underlying data.
* **Stakeholder Trust:** Transparency in credit scoring models builds trust among customers, investors, and regulators. It helps explain adverse decisions (e.g., loan rejection) and promotes fairness.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In our dataset, we do not have a direct label indicating whether a customer "defaulted" on a loan. A proxy variable is **necessary** because a machine learning model requires a target variable (what we want to predict). Without a direct default label, we must construct a substitute that, to the best of our knowledge, correlates strongly with actual credit default behavior. This proxy will serve as our "dependent variable" for model training.

**Potential Business Risks of Making Predictions Based on a Proxy:**

* **Proxy Imperfection (Bias):** The proxy may not perfectly capture the true definition of default. For example, if we define "bad" as customers with multiple fraud incidents, this might miss customers who default due to inability to pay (e.g., job loss) but never engaged in fraud. This introduces bias into the model's predictions.
* **Misclassification Costs:**
    * **False Positives (Type I Error - "Good" customers labeled "Bad"):** We might incorrectly label creditworthy customers as high risk. This leads to **lost revenue** (missed lending opportunities, lower interest income), **reduced market share**, and potential **customer dissatisfaction** or churn if deserving customers are denied credit or offered unfavorable terms.
    * **False Negatives (Type II Error - "Bad" customers labeled "Good"):** We might incorrectly label high-risk customers as low risk. This leads to **financial losses** from actual defaults, **increased operational costs** for debt collection, and **reputational damage**.
* **Model Drift:** The relationship between the proxy variable and true default behavior might change over time due to shifts in economic conditions, customer behavior, or fraud patterns. This could lead to the model becoming less accurate and making increasingly unreliable predictions without a mechanism for recalibration.
* **Regulatory Scrutiny:** Regulators may question the validity and robustness of a model built on a proxy, especially if the proxy definition is not clearly justified or if its correlation with actual default is weak.
* **Ethical Concerns/Fairness:** If the proxy implicitly correlates with protected characteristics (e.g., socioeconomic status, indirectly linked to demographic groups), the model might inadvertently perpetuate bias, leading to unfair lending practices. This is where careful proxy definition and monitoring are crucial.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

In a regulated financial context, the choice between model complexity and interpretability involves significant trade-offs:

**Simple, Interpretable Models (e.g., Logistic Regression with WoE, Decision Trees):**

* **Pros:**
    * **High Interpretability:** Easy to understand how each feature influences the prediction. For Logistic Regression with WoE, the Weights of Evidence provide clear insights into the predictive power and direction of categories. This directly aids regulatory compliance, model validation, and business understanding.
    * **Transparency:** Stakeholders can readily follow the logic of the model, fostering trust.
    * **Regulatory Acceptance:** Often preferred by regulators due to their transparency and ease of auditing.
    * **Explainability for Customers:** Easier to explain why a loan decision was made, which is often a regulatory requirement (e.g., adverse action notices).
    * **Less Prone to Overfitting (generally):** Simpler models have fewer parameters and are less likely to overfit noisy data.
* **Cons:**
    * **Lower Predictive Performance (Potentially):** May not capture complex, non-linear relationships in the data as effectively as complex models, potentially leading to lower accuracy or F1 scores.
    * **Requires More Feature Engineering:** To compensate for their simplicity, these models often require more extensive and thoughtful feature engineering (e.g., interaction terms, polynomial features) to achieve competitive performance.
    * **Assumptions:** Logistic Regression assumes a linear relationship between log-odds of the outcome and features (or WoE transformed features), which might not always hold true for real-world data.

**Complex, High-Performance Models (e.g., Gradient Boosting Machines like XGBoost, LightGBM; Neural Networks):**

* **Pros:**
    * **Higher Predictive Performance:** Often achieve superior accuracy, F1-score, and AUC due to their ability to model highly complex, non-linear relationships and interactions between features.
    * **Automated Feature Interaction Discovery:** Can implicitly learn complex feature interactions without explicit manual engineering.
    * **Robustness to Missing Values/Outliers:** Some algorithms (e.g., tree-based ensembles) are inherently more robust to outliers and can handle certain types of missing values.
* **Cons:**
    * **Low Interpretability ("Black Box"):** Difficult to understand *why* a specific prediction was made. While tools like SHAP and LIME can provide post-hoc explanations, they don't offer the inherent transparency of simpler models.
    * **Regulatory Skepticism:** Regulators are often cautious about black-box models due to concerns about fairness, bias, and the inability to fully audit their decision-making process.
    * **Challenging Debugging:** Diagnosing issues in complex models can be very difficult.
    * **Higher Computational Cost:** Training and deploying can be more resource-intensive.
    * **Higher Risk of Overfitting:** More parameters mean they can easily overfit if not properly regularized and validated.

**Key Trade-offs Summary:**

| Feature               | Simple, Interpretable Models (e.g., LR with WoE) | Complex, High-Performance Models (e.g., Gradient Boosting) |
| :-------------------- | :----------------------------------------------- | :--------------------------------------------------------- |
| **Performance** | Moderate to Good                                 | Excellent                                                  |
| **Interpretability** | High                                             | Low (Requires post-hoc techniques)                         |
| **Regulatory Ease** | High (Preferred)                                 | Low (High Scrutiny)                                        |
| **Feature Engineering** | More manual effort needed                        | Less manual effort for interactions                        |
| **Debugging** | Easier                                           | Difficult                                                  |
| **Risk of Bias** | Easier to detect and mitigate                    | Harder to detect and mitigate (implicit biases)            |
| **Computational Cost** | Lower                                            | Higher                                                     |

In a regulated financial context, there's often a strong preference for interpretability due to compliance, auditing, and trust requirements. 
A common strategy is to start with simpler, interpretable models, and if their performance is insufficient, then move to more complex models, always 
accompanied by robust explainability (XAI) techniques and rigorous validation processes. The optimal choice often depends on the specific regulatory 
environment, the acceptable level of risk, and the performance gains achievable by the more complex model.