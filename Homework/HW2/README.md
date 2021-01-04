# NLP-Sentiment Analysis on Amazon Reviews - Part 2
## Extract imperative sentences and question sentences that contain adjective phrases

### Introduction
<p>
	The analysis is conducted on the review contents from Amazon Product Data provided by Julian McAuley at http://jmcauley.ucsd.edu/data/amazon/. This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014. It includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs).
</p>

### Requirement
<p>
	For this project, only the 5-core subsets of the category "Clothing, Shoes and Jewelry" will be used. The 5-core subsets mean that all users and items in the dataset have at least 5 reviews.
	The data is downloaded and stroed as a text file named "clothing shoes jewelry.txt".
	Extract only the text contents and store it to a new text file named "reviews.txt".
	The imperative sentences and question sentences will be extracted, which contain adjective phrases. A descriptive statistics as well as the unigram and bigram frequency analysis on them will be conducted after.
</p>

### How to Implement the Extraction
<ol>
	<li>Went through some grammar websites to design several grammar rules.</li>
	<li>Extracted all the sentences end with question mark first, import StandfordCoreNLP to analyse the data and apply the grammar rule for question sentences on them.</li>
	<li>Extracted all the sentences end with exclamation mark, and apply the grammar rule for imperative sentences on them.</li>
</ol>
