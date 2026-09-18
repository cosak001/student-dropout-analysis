# Student Dropout Analysis

**Which first-year students are most likely to drop out, and is the college's first-year seminar helping?**

Analysis of 600 first-year student records from a case-study college, covering observed retention patterns, predictive modeling, and an interactive Tableau dashboard built for administrators.

**[View the interactive dashboard →](PASTE_YOUR_TABLEAU_LINK_HERE)**

![Dropout Findings dashboard](images/dropout_findings.png)

**Tools** Tableau, SPSS, Excel
**Methods** Exploratory analysis, logistic regression, multilayer perceptron, dashboard design

---

## Background

One in five first-year students at the college did not return after their first year. Administrators wanted to know which students were most at risk, and whether the voluntary first-year seminar was improving retention.

## Data

600 first-year student records from Spring 2026. Each record contains high school GPA, units enrolled, age, gender, registration status, commuter status, first-year seminar attendance, and whether the student dropped out. No missing values. This is a course case dataset rather than real institutional records.

## Findings

**High school GPA was the strongest predictor of dropout.** Students who left averaged a 2.66 GPA against 2.97 for those who stayed. Dropout fell steadily as GPA rose, from 33% among students below a 2.5 to 7% among students at 3.5 and above.

**Heavier course loads were associated with more dropout, not less.** Students who left carried 13.5 units on average against 12.2 for those who stayed. Dropout climbed from 11% among students taking fewer than 9 units to 29% among those taking 16 to 18.

**Commuters were retained at higher rates.** 13% of commuters dropped out, against 23% of non-commuters.

**Seminar attendees dropped out more often, but the comparison is confounded.** 27% of attendees left, against 16% of non-attendees. A second logistic regression predicting attendance showed that students with lower GPAs were significantly more likely to attend, so attendees were already at higher risk before the seminar began. Because attendance was voluntary, the difference cannot be read as an effect of the seminar.

**The highest-risk group is students with both risk factors.** Estimated dropout risk reaches 51% for students below a 2.5 GPA carrying 16 to 18 units, against 2% for students at 3.5 and above taking fewer than 9.

## Recommendations

Assign seminar seats at random for the next cohort, so the program can be evaluated against a comparable control group instead of a self-selected one.

Use GPA and course load together to flag students for early outreach, since either factor alone understates risk for students who carry both.

Lower the classification threshold if the models are used operationally. At the default cutoff the model identifies few of the students who actually leave, and for outreach the cost of a missed at-risk student outweighs the cost of an unnecessary check-in.

## Methods

Group comparisons and distribution analysis established the observed patterns. A logistic regression on four predictors modeled dropout, a second logistic regression on seminar attendance tested for selection effects, and a multilayer perceptron was fitted on the same predictors for comparison. Modeling was performed in SPSS; the dashboards were built in Tableau.

The final dropout model.

| Predictor | Coefficient | Odds ratio | Significance |
|---|---|---|---|
| High school GPA | −1.621 | 0.20 | p < .001 |
| Units enrolled | 0.147 | 1.16 | p < .001 |
| Commuter status | −0.800 | 0.45 | p = .003 |
| Seminar attendance | 0.673 | 1.96 | p = .003 |

Nagelkerke R² = .202. Hosmer–Lemeshow p = .114, showing no evidence of poor fit.

## Limitations

Overall accuracy was roughly 81%, but 80% of students were retained, so a model predicting that nobody drops out would score nearly as well. At the default 0.5 cutoff the logistic model identified 13% of the students who actually left. Accuracy is therefore the wrong headline metric for this problem, and threshold choice matters more than model choice.

The neural network was run without a fixed random seed and on a different train/test split from the logistic model, so its results are neither exactly reproducible nor directly comparable.

The GPA and course load groupings used in the dashboards are display bands chosen for readability, not validated risk thresholds.

Estimated risk values assume a non-commuting student who did not attend the seminar.

## Dashboard

Two views, published on Tableau Public.

**Dropout Findings** presents the observed differences in dropout rates across seminar attendance, commuter status, GPA, and course load, with the selection-bias caveat stated alongside the seminar result.

**Student Dropout Risk Patterns** adds model-estimated risk across every combination of GPA band and course load, with observed rates and group sizes available on hover so estimates can be checked against what actually happened.

![Student Dropout Risk Patterns dashboard](images/risk_patterns.png)

## Repository structure

```
student-dropout-analysis
├── README.md
├── data
│   ├── student_data.csv                 600 student records
│   └── MontStMichelSpring2026.xlsx      source spreadsheet
├── tableau
│   └── StudentDropoutAnalysis.twbx      workbook with both dashboards
├── docs
│   └── report.pdf                       full written analysis
├── spss
│   └── FINAL_OUTPUT_GROUP_PROJECT.spv   model output for all six models
└── images
    ├── dropout_findings.png
    └── risk_patterns.png
```

## Contributions

ISDS 415 group project, Cal State Fullerton, Spring 2026. Each team member independently built logistic regression and neural network models in SPSS, and we compared results across analysts as a check on the findings. I later redesigned and rebuilt the Tableau dashboards on my own, including new calculated fields, corrected dropout rate calculations, and all visual design.
