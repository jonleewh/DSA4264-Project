DSA4264 Technical Report
====================

There is a maximum word count of 3000 words – you must learn to prioritise the most critical content. Feel free to use charts or tables to illustrate your point. The only restriction is that you must write in clear prose.


# 1. Context

In this section, you need to explain why your team chose to do this project. There are many possible reasons for this – huge impact, urgently needed, boss said so.

In this section, you should explain how this project came about. Retain all relevant details about the project’s history and context, especially if this is a continuation of a previous project. 

If there are any slide decks or email threads that started before this project, you should include them as well.


# 2. Scope

Scope helps to sharpen the focus of your work onto a specific technical problem that, if solved, helps to address a key business problem.

## 2.1 Problem

Write this from a business perspective:

- What is the problem that the business unit faces? Be specific about who faces the problem, how frequently it occurs, and how it affects their ability to meet their desired goals.

- What is the significance or impact of this problem? Provide tangible metrics that demonstrate the cost of not addressing this problem. 

- Why is data science / machine learning the appropriate solution to the problem?


## 2.2 Success Criteria

- Specify at least 2 business and/or operational goals that will be met if this project is successful.

- Business goals directly relate to the business’s objectives, such as reduced fraud rates or improved customer satisfaction.

- Operational goals relate to the system’s needs, such as better reliability, faster operations, etc.


## 2.3 Assumptions

- Set out the key assumptions for this data science project that, if changed, will affect the problem statement, success criteria, or feasibility. You do not need to detail out every single assumption if the expected impact is not significant.

- For example, if we are building an automated fraud detection model, one important assumption may be whether there is enough manpower to review each individual decision before proceeding with it.


# 3. Methodology

You need to provide the right level of details depending on how important or controversial the decision may be:

- Important / controversial decisions: Give a clear and detailed explanation, share alternatives considered in the annex

- Unimportant decisions: Just say what was decided

## 3.1 Technical Assumptions

Set out the assumptions that are directly related to your model development process. Some general categories include:

- How to define certain terms as variables

- What features are available / not available

- What kind of computational resources are available to you (ie on-premise vs cloud, GPU vs CPU, RAM availability)

- What the key hypotheses of interest are

- What the data quality is like (especially if incomplete / unreliable)

## 3.2 Data

Provide a clear and detailed explanation of how your data is collected, processed, and used. Some specific parts you should explain are:

- Collection: What datasets did you use and how are they collected?

- Cleaning: How did you clean the data? How did you treat outliers or missing values?

- Features: What feature engineering did you do? Was anything dropped?

- Splitting: How did you split the data between training and test sets?

## 3.3 Experimental Design

Clearly explain the key steps of your model development process, such as:

- Algorithms: Which ML algorithms did you choose to experiment with, and why?

- Evaluation: Which evaluation metric did you optimise and assess the model on? Why is this the most appropriate?

- Training: How did you arrive at the final set of hyperparameters? How did you manage imbalanced data or regularisation?


# 4. Findings

The final section is generally the longest as it includes the results and discussion about the experimentation that was done.

Beyond just reporting the results, you need to situate them in context of the business problem and how they satisfy (or not) the business goals that you set out in the earlier section.

## 4.1 Results

Report the results from your experiments in a **summary table**, keeping only the most relevant results for your experiment (ie your best model, and two or three other options which you explored). You should also briefly explain the summary table and highlight key results.

Interpretability methods like LIME or SHAP should also be reported here, using the appropriate tables or charts.

## 4.2 Discussion

What is important is not just the results, but also how to interpret them correctly and how it impacts the business problem.

Use this sub-section to discuss what the results mean for the business user – specifically how the technical metrics translate into business value and costs, and whether this has sufficiently addressed the business problem.

You should also discuss or highlight other important issues like interpretability, fairness, and deployability.


## 4.3 Recommendations

Having discussed the key results, it’s now time to make a recommendation on what to do next – often this is either to deploy the model or to close off this project

For most projects, what to do next is either to deploy the model into production or to close off this project and move on to something else. Reasoning about this involves understanding the business value, and the potential IT costs of deploying and integrating the model.

Other things you can recommend would typically relate to data quality and availability, or other areas of experimentation that you did not have time or resources to do this time round.

