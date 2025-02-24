# Bank-Marketing-Segmentation - Supervised ML
---

![Microsoft Excel](https://img.shields.io/badge/Microsoft_Excel-217346?style=for-the-badge&logo=microsoft-excel&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Markdown](https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor=white)
![Microsoft Office](https://img.shields.io/badge/Microsoft_Office-D83B01?style=for-the-badge&logo=microsoft-office&logoColor=white)
![Microsoft Word](https://img.shields.io/badge/Microsoft_Word-2B579A?style=for-the-badge&logo=microsoft-word&logoColor=white)
![Microsoft PowerPoint](https://img.shields.io/badge/Microsoft_PowerPoint-B7472A?style=for-the-badge&logo=microsoft-powerpoint&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![R](https://img.shields.io/badge/R-%23276DC3.svg?logo=r&logoColor=white&style=for-the-badge)

---

## Problem Overview


This dataset consists of tv shows and movies available on Amazon Prime as of 2019. The dataset is collected from Flixable which is a third-party Amazon Prime search engine.

This project provides an overview of the results derived from analysing the bank's dataset and offers conclusions based on the investigation.

CONTEXT: With rapid technological advancements, competition in the banking industry has increased, making it increasingly difficult to attract new customers. To address this, the bank's marketing team has launched several campaigns promoting term deposit subscriptions. However, acquiring and retaining customers can be costly and resource-intensive. To mitigate these costs, the bank aims to focus its promotional efforts on “high-value customers” — those most likely to open deposit accounts. Our analysis focuses on customer segmentation to identify groups most likely to subscribe to a deposit account. We employ a variety of binary classification models to achieve this goal.

In this project, required to do:

- Background
- Business Problem
- Proposed Solution
- Method of Analysis
- Results and Interpretation
- Strategy and Recommendations

---

## Project Summary

This dataset consists of tv shows and movies available on Amazon Prime as of 2021. The dataset is collected from Flixable which is a third-party Amazon Prime search engine.

In 2021, they released an interesting report which shows that the number of TV shows on Amazon Prime has nearly tripled since 2010. The streaming service’s number of movies has decreased by more than 2,000 titles since 2010, while its number of TV shows has nearly tripled. It will be interesting to explore what all other insights can be obtained from the same dataset.

Initially I have start with understanding the dataset, then I clean the data to make analysis ready.

Explore the data and understand the behaviour of the same.

Then I have prepare the dataset for creating clusters by various parameters wherein I can remove stop words, white spaces, numbers etc. so that I can get important words and based on that i shall form clusters.

Later I have used the silhouette method and k-means elbow method to find optimal number of clusters and built recommender system by cosine similarity and recommended top ten movies.

---

## Conclusion

Key insights of the key attributes of high-value customers from the project include: 
- Customers who are older, work in management, and have a high school education, professional certification, or university degree are more likely to subscribe.
  
- Customers who are younger, work in admin, blue-collar jobs, as entrepreneurs, housemaids, self-employed, or in services, and have only a basic six-year education, are less likely to subscribe.
  
- Customers are more likely to subscribe when the consumer confidence index is above -41.4 and the Euribor 3-month rate is below 1.26.

- A customer is significantly more likely to subscribe if their interaction duration from the last contact exceeds 162.4 seconds.

- Other key attributes of high-value customers include fewer days since the last contact (pdays), a higher number of contracts from both current and previous campaigns, and whether they subscribed to the previous campaign.

What can the Bank do with this?
To maximise the effectiveness of this data, the team can take several steps:
- Create programs related to the  high-value customer’s characteristics 
- Design campaigns that specifically attracts high-value customers
- Launch campaigns when the customers' economic indicators, such as the consumer confidence index and Euribor rate, fall within the optimal range mentioned above.

## Recomendations
Three recommendations were suggested for the marketing team’s next project:
- Complete the development of programs targeted at high-value customers and begin advertising them immediately.
- Only promote programs and campaigns (from prior projects) to customers whose current economic indicators fall within the ideal range.
- Ensure all three key steps are finalised before launching the next campaign.
The potential costs associated with these recommendations include increased expenses, longer timelines, and higher labour demands.

As a result, the most and least feasible recommendations have been identified for the marketing team to consider:
- Most Feasible Recommendation: Release the programs already developed for customers likely to subscribe, based on their characteristics. This approach is more practical because the effectiveness of these strategies is still unknown. By rolling out this initial step and gathering data, the marketing team can implement the most effective strategies and make adjustments as needed. If the data reveals that the strategies are ineffective, the team can refine them before designing campaigns specifically targeting high-value customers.
- Least Feasible Recommendation: Implement all strategies at once for the next campaign. This option is less feasible due to the significant costs, time, and labour involved. Additionally, it is uncertain when the campaign can be launched, as it depends on unpredictable factors like when customers’ economic indicators reach the desired range.

---

## Author

- [Eileen Ip](https://www.linkedin.com/in/eileen-ip/)
