# NLP

Natural Language Processing (NLP) is a sub-field of machine learning that attempts to make human languages (e.g. English, French, Mandarin) machine readable and interpretable. 
NLP poses notoriously difficult challenges, including the ambiguity of written text, the variability of meaning given context, the use of slang, idioms and abbreviations, the use of metaphor and analogy, and deciphering the true intent of a writer.

The nuts and bolts of NLP include all of the following:
1. Text extraction in a conistent and reliable format from multiple channels across multiple sources of truth.
2. Text sanitization including removal of artifacts from web sources such as url-encoding, HTML markup, CSS tags and inline javascript.
3. Lemmatization and stemming for normalization.
4. Part-of-speech tagging, named-entity recognition, dependency parsing or other advanced linguistic models.
5. Topic modeling, document clustering or other high-level abstractions.

## Challenge
Let's suppose you never read the Harry Potter series (my sincerest apologies).  
Now suppose you want to get the gist without spending dozens of hours watching the movies or hundreds of hours reading the books (although you really should).  
How could you leverage natural language processing to quickly extract topics, themes, or plotlines?  

Specifically, attempt to algorithmically assign a title and synopsis to each document (in this case let's consider each chapter as a document);
and then repeat the process at the level of topics (that is, identify topics from the corpus of text, and assign a title and synopsis to each topic).

Assume steps 1 & 2 have been completed for you with a robust ETL pipeline.  
Tackle steps 3, 4 and 5, with emphasis on step 5.  
The number of topics and how you present your findings is entirely up to you.  
One catch -- please refrain from using the python library LDAvis -- we want to see how you build from the ground up.  
That said, you may use any other machine learning or NLP library for underlying computations and/or transfer-learning. 

## Guidelnes

1. Do not copy source code from anywhere. We want to see how you would approach this problem yourself.
2. Your code should be written primarily in Python.
3. Your code should run.
4. Don't worry about fancy optimizations, edge cases, bells or whistles.
5. Timebox your efforts to a maximum of 8 hours.
6. The result of your efforts should be a pull-request to this repository. 

## Considerations
Finally, one of the most important considerations for a full-stack data scientist is the ability to write production software. The best model is only as good as its deployment. Please use this opportunity to:

1. Demonstrate your ability to write clean code that is robust, modular and well structured.
3. Demonstrate software engineering best practices including self-documenting code, test-driven development, atomic-commits and source control.

Oh and have fun!
