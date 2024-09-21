# Pagerank Project

In this project, you will create a simple search engine for the website <https://www.lawfareblog.com>.
This website provides legal analysis on US national security issues.
You will use pagerank to return only the most important results from this website in your search engine.

**Due date:** Sunday, 22 September at midnight

**Late Policy:** You lose $2^{(i-1)}$ points, where i is the number of days late.

<!--
**Computation:**
This project has low computational requirements.
You should be able to complete it on your own laptops.
-->

**Collaboration Policy:**
Do whatever will help you learn,
but be an adult.
You may talk to other students and use Google/ChatGPT.
Recall that you will have an in-person oral exam on this material and the exam is worth many more points.
The main purpose of this project is to help prepare you for the exam.

## Background

**Data:**

The `data` folder contains two files that store example "web graphs".
The file `small.csv.gz` contains the example graph from the *Deeper Inside Pagerank* paper.
This is a small graph, so we can manually inspect the contents of this file with the following command:
```
$ zcat data/small.csv.gz
source,target
1,2
1,3
3,1
3,2
3,5
4,5
4,6
5,6
5,4
6,4
```

> **Recall:**
> The `cat` terminal command outputs the contents of a file to stdout, and the `zcat` command first decompressed a gzipped file and then outputs the decompressed contents.
>
> In python, we can use the built-in `gzip` module to access gzipped files.
> The following python code is equivalent to the bash code above:
>
> ```
> >>> import gzip
> >>> fin = gzip.open('data/small.csv.gz', mode='rt')
> >>> print(fin.read())
> source,target
> 1,2
> 1,3
> 3,1
> 3,2
> 3,5
> 4,5
> 4,6
> 5,6
> 5,4
> 6,4
> ```
>
> There are many terminal commands throughout these instructions.
> If you haven't used the terminal before, and so these commands are unfamiliar, that's okay.
> I'd be happy to explain them in office hours,
> or there are many tutors in the QCL available who can help.
> (There are no tutors for this class specifically, but anyone who has taken CSCI046 or CSCI133 with me will be able to help with the terminal.)
>
> Furthermore, you don't "need" to understand the terminal commands in detail,
> since you are not required to run these commands or to create your own.
> The important part is to understand the English language description of what the commands are doing,
> and to understand that this is just how I computed what the English language text is describing.

As you can see, the graph is stored as a CSV file.
The first line is a header,
and each subsequent line stores a single edge in the graph.
The first column contains the source node of the edge and the second column the target node.
The file is assumed to be sorted alphabetically.

The second data file `lawfareblog.csv.gz` contains the link structure for the lawfare blog.
Let's take a look at the first 10 of these lines:
```
$ zcat data/lawfareblog.csv.gz | head
source,target
www.lawfareblog.com/,www.lawfareblog.com/topic/interrogation
www.lawfareblog.com/,www.lawfareblog.com/upcoming-events
www.lawfareblog.com/,www.lawfareblog.com/
www.lawfareblog.com/,www.lawfareblog.com/our-comments-policy
www.lawfareblog.com/,www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
www.lawfareblog.com/,www.lawfareblog.com/topic/lawfare-research-paper-series
www.lawfareblog.com/,www.lawfareblog.com/topic/book-reviews
www.lawfareblog.com/,www.lawfareblog.com/documents-related-mueller-investigation
www.lawfareblog.com/,www.lawfareblog.com/topic/international-law-loac
```
You can see that in this file, the node names are URLs.
Semantically, each line corresponds to an HTML `<a>` tag that is contained in the source webpage and links to the target webpage.

We can use the following command to count the total number of links in the file:
```
$ zcat data/lawfareblog.csv.gz | wc -l
1610789
```
Since every link corresponds to a non-zero entry in the $P$ matrix,
this is also the value of $\text{nnz}(P)$.
(Technically, we should subtract 1 from this value since the `wc -l` command also counts the header line, not just the data lines.)

To get the dimensions of $P$, we need to count the total number of nodes in the graph.
The following command achieves this by: decompressing the file, extracting the first column, removing all duplicate lines, then counting the results.
```
$ zcat data/lawfareblog.csv.gz | cut -f1 -d, | uniq | wc -l
25761
```
This matrix is large enough that computing matrix products for dense matrices takes several minutes on a single CPU.
Fortunately, however, the matrix is very sparse.
The following python code computes the fraction of entries in the matrix with non-zero values:
```
>>> 1610788 / (25760**2)
0.0024274297384360172
```
Thus, by using sparse matrix operations, we will be able to speed up the code significantly.

**Code:**

The `pagerank.py` file contains code for loading the graph CSV files and searching through their nodes for key phrases.
For example, you can perform a search for all nodes (i.e. urls) that mention the string `corona` with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --search_query=corona
```

> **NOTE:**
> It will take about 10 seconds to load and parse the data files.
> All the other computation happens essentially instantly.

Currently, the pagerank of the nodes is not currently being calculated correctly, and so the webpages are returned in an arbitrary order.
Your task in this assignment will be to fix these calculations in order to have the most important results (i.e. highest pagerank results) returned first.

## Task 1: the power method

Implement the `WebGraph.power_method` function in `pagerank.py` for computing the pagerank vector by fixing the [`FIXME: Task 1` annotation](https://github.com/mikeizbicki/cmc-csci145-math166/blob/81ed5d2b75f5bc23b8de93805c29321ab431ed9b/topic01_computation_pagerank/project/pagerank.py#L144).

> **NOTE:**
> The power method is the only data mining algorithm you will implement in class.
> You are implementing it because there are no standard library implementations available.
> Why?
> 1. The runtime is heavily dependent on the data structures used to store the graph data.
>    Different applications will need to use different data structures.
> 1. It is "trivial" to implement.
>    My solution to this homework is <10 lines of code.

**Part 1:**

To check that your implementation is working,
you should run the program on the `data/small.csv.gz` graph.
For my implementation, I get the following output.
```
$ python3 pagerank.py --data=data/small.csv.gz --verbose
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=2.5629e-01
DEBUG:root:i=1 residual=1.1841e-01
DEBUG:root:i=2 residual=7.0701e-02
DEBUG:root:i=3 residual=3.1815e-02
DEBUG:root:i=4 residual=2.0497e-02
DEBUG:root:i=5 residual=1.0108e-02
DEBUG:root:i=6 residual=6.3716e-03
DEBUG:root:i=7 residual=3.4228e-03
DEBUG:root:i=8 residual=2.0879e-03
DEBUG:root:i=9 residual=1.1750e-03
DEBUG:root:i=10 residual=7.0131e-04
DEBUG:root:i=11 residual=4.0321e-04
DEBUG:root:i=12 residual=2.3800e-04
DEBUG:root:i=13 residual=1.3812e-04
DEBUG:root:i=14 residual=8.1083e-05
DEBUG:root:i=15 residual=4.7251e-05
DEBUG:root:i=16 residual=2.7704e-05
DEBUG:root:i=17 residual=1.6164e-05
DEBUG:root:i=18 residual=9.4778e-06
DEBUG:root:i=19 residual=5.5066e-06
DEBUG:root:i=20 residual=3.2042e-06
DEBUG:root:i=21 residual=1.8612e-06
DEBUG:root:i=22 residual=1.1283e-06
DEBUG:root:i=23 residual=6.1907e-07
INFO:root:rank=0 pagerank=6.6270e-01 url=4
INFO:root:rank=1 pagerank=5.2179e-01 url=6
INFO:root:rank=2 pagerank=4.1434e-01 url=5
INFO:root:rank=3 pagerank=2.3175e-01 url=2
INFO:root:rank=4 pagerank=1.8590e-01 url=3
INFO:root:rank=5 pagerank=1.6917e-01 url=1
```
Yours likely won't be identical (due to minor implementation details and weird floating point issues), but it should be similar.
In particular, the ranking of the nodes/urls should be the same order.

> **NOTE:**
> The `--verbose` flag causes all of the lines beginning with `DEBUG` to be printed.
> By default, only lines beginning with `INFO` are printed.

> **NOTE:**
> There are no automated test cases to pass for this assignment.
> Test cases for algorithms involving floating point computations are hard to write and understand.
> Minor-seeming implementations details can have large impacts on the final result.
> These software engineering issues are beyond the scope of this class.
>
> Instructions for how I will grade your homework are contained in the [submission section](#submission) at the end of this document.

**Part 2:**

The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter.
If this argument is used, then the program returns all nodes that match the query string sorted according to their pagerank.
Essentially, this gives us the most important pages related to our query.

Again, you may not get the exact same results as me,
but you should get similar results to the examples I've shown below.
Verify that you do in fact get similar results.

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=8.9224e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.0390e-04 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=6.9153e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=6.7041e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=6.6256e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=6.5046e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=6.3620e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=6.1248e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=6.0187e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
INFO:root:rank=0 pagerank=5.7826e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=5.2338e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=2 pagerank=5.1297e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=3 pagerank=4.6599e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=4 pagerank=4.5934e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=4.3071e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=4.0935e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=3.7591e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=3.4509e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=3.4484e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
INFO:root:rank=0 pagerank=4.5746e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=1 pagerank=4.4174e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=2 pagerank=2.6928e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=1.9391e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=4 pagerank=1.5452e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=5 pagerank=1.5357e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=7 pagerank=1.4221e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
```

**Part 3:**

The webgraph of lawfareblog.com (i.e. the $P$ matrix) naturally contains a lot of structure.
For example, essentially all pages on the domain have links to the root page <https://lawfareblog.com/> and other "non-article" pages like <https://www.lawfareblog.com/topics> and <https://www.lawfareblog.com/subscribe-lawfare>.
These pages therefore have a large pagerank.
We can get a list of the pages with the largest pagerank by running

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
```

Most of these pages are not very interesting, however, because they are not articles,
and usually when we are performing a web search, we only want articles.

This raises the question: How can we find the most important articles filtering out the non-article pages?
The answer is to modify the $P$ matrix by removing all links to non-article pages.

This raises another question: How do we know if a link is a non-article page?
Unfortunately, this is a hard question to answer with 100% accuracy,
but there are many methods that get us most of the way there.
One easy to implement method is to compute what's called the "in-link ratio" of each node (i.e. the total number of edges with the node as a target divided by the total number of nodes),
and then remove nodes from the search results with too-high of a ratio.
The intuition is that non-article pages often appear in the menu of a webpage, and so have links from almost all of the other webpages;
but article-webpages are unlikely to appear on a menu and so will only have a small number of links from other webpages.
The `--filter_ratio` parameter causes the code to remove all pages that have an in-link ratio larger than the provided value.

Using this option, we can estimate the most important articles on the domain with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```
Notice that the urls in this list look much more like articles than the urls in the previous list.

When Google calculates their $P$ matrix for the web,
they use a similar (but much more complicated) process to modify the $P$ matrix in order to reduce spam results.
The exact formula they use is a jealously guarded secret that they update continuously.

In the case above, notice that we have accidentally removed the blog's most popular article (<https://www.lawfareblog.com/snowden-revelations>).
The blog editors believed that Snowden's revelations about NSA spying are so important that they directly put a link to the article on the menu.
So every single webpage in the domain links to the Snowden article,
and our "anti-spam" `--filter-ratio` argument removed this article from the list.
In general, it is a challenging open problem to remove spam from pagerank results,
and all current solutions rely on careful human tuning and still have lots of false positives and false negatives.

**Part 4:**

Recall from the reading that the runtime of pagerank depends heavily on the eigengap of the $\bar{\bar P}$ matrix,
and that this eigengap is bounded by the alpha parameter.

Run the following four commands:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
```
You should notice that the last command takes considerably more iterations to compute the pagerank vector.
(My code takes 685 iterations for this call, and about 10 iterations for all the others.)

This raises the question: Why does the second command (with the `--alpha` option but without the `--filter_ratio`) option not take a long time to run?
The answer is that the $P$ graph for <https://www.lawfareblog.com> naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence.

Changing the value of alpha also gives us very different pagerank rankings.
For example, 
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --alpha=0.99999
INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=3.1755e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

Which of these rankings is better is entirely subjective,
and the only way to know if you have the "best" alpha for your application is to try several variations and see what is best.

> **NOTE:**
> It should be "obvious" to you that large alpha values imply that the structure of the webgraph has more influence on the final result,
> and small alpha values ignore the structure of the webgraph.
> Recall that the word "obvious" means that it follows directly from the definition,
> but you may still need to sit and meditate on the definition for a long period of time.

If large alphas are good for your application, you can see that there is a trade-off between quality answers and algorithmic runtime.
We'll be exploring this trade-off more formally in class over the rest of the semester.

## Task 2: the personalization vector

The most interesting applications of pagerank involve the personalization vector.
Implement the `WebGraph.make_personalization_vector` function so that it outputs a personalization vector tuned for the input query.
The pseudocode for the function is:
```
for each index in the personalization vector:
    get the url for the index (see the _index_to_url function)
    check if the url satisfies the input query (see the url_satisfies_query function)
    if so, set the corresponding index to one
normalize the vector
```

**Part 1:**

The command line argument `--personalization_vector_query` will use the function you created above to augment your search with a custom personalization vector.
If you've implemented the function correctly,
you should get results similar to:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

Notice that these results are significantly different than when using the `--search_query` option:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --search_query='corona'
INFO:root:rank=0 pagerank=8.1320e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=1 pagerank=7.7908e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=2 pagerank=5.2262e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=3 pagerank=3.9584e-03 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=4 pagerank=3.8114e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=5 pagerank=3.3973e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=6 pagerank=3.3633e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus
INFO:root:rank=7 pagerank=3.3557e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=3.2160e-03 url=www.lawfareblog.com/congress-needs-coronavirus-failsafe-its-too-late
INFO:root:rank=9 pagerank=3.1036e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
```

Which results are better?
Again, that depends on what you mean by "better."
With the `--personalization_vector_query` option,
a webpage is important only if other coronavirus webpages also think it's important;
with the `--search_query` option,
a webpage is important if any other webpage thinks it's important.
You'll notice that in the later example, many of the webpages are about Congressional proceedings related to the coronavirus.
From a strictly coronavirus perspective, these are not very important webpages.
But in the broader context of national security, these are very important webpages.

Google engineers spend TONs of time fine-tuning their pagerank personalization vectors to remove spam webpages.
Exactly how they do this is another one of their secrets that they don't publicly talk about.

**Part 2:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
```
You can see that there are many urls about concepts that are obviously related like "covid", "clinical trials", and "quarantine",
but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions.
Both of these articles are highly relevant to coronavirus discussions,
but a simple keyword search for corona or related terms would not find these articles.
The vast majority of industry data mining work is finding clever uses of standard algorithms.

<!--
**Part 3:**

Select another topic related to national security.
You should experiment with a national security topic other than the coronavirus.
For example, find out what articles are important to the `iran` topic but do not contain the word `iran`.
Your goal should be to discover what topics that www.lawfareblog.com considers to be related to the national security topic you choose.
-->

## Submission

1. Create a new repo on github (not a fork of this repo).
    Ensure that all of the project files are copied from this folder into your new repo.

1. As you complete the tasks above:
    Run the corresponding commands below, and paste their output into the code blocks.
    Please ensure correct markdown formatting.
   
   Task 1, part 1:
   ```
   $ python3 pagerank.py --data=data/small.csv.gz --verbose
   DEBUG:root:computing indices
   DEBUG:root:computing values
   DEBUG:root:i=0 residual=0.6277832984924316
   DEBUG:root:i=1 residual=0.11841224879026413
   DEBUG:root:i=2 residual=0.07070129364728928
   DEBUG:root:i=3 residual=0.03181544691324234
   DEBUG:root:i=4 residual=0.02049662359058857
   DEBUG:root:i=5 residual=0.010108369402587414
   DEBUG:root:i=6 residual=0.006371521856635809
   DEBUG:root:i=7 residual=0.003422821406275034
   DEBUG:root:i=8 residual=0.00208795303478837
   DEBUG:root:i=9 residual=0.0011749333934858441
   DEBUG:root:i=10 residual=0.0007013526046648622
   DEBUG:root:i=11 residual=0.0004031880816910416
   DEBUG:root:i=12 residual=0.00023802061332389712
   DEBUG:root:i=13 residual=0.00013810257951263338
   DEBUG:root:i=14 residual=8.112059731502086e-05
   DEBUG:root:i=15 residual=4.7194993385346606e-05
   DEBUG:root:i=16 residual=2.7686633984558284e-05
   DEBUG:root:i=17 residual=1.616942790860776e-05
   DEBUG:root:i=18 residual=9.459256943955552e-06
   DEBUG:root:i=19 residual=5.5136310948000755e-06
   DEBUG:root:i=20 residual=3.2534051115362672e-06
   DEBUG:root:i=21 residual=1.8578108438305208e-06
   DEBUG:root:i=22 residual=1.1294450814602897e-06
   DEBUG:root:i=23 residual=6.332555244625837e-07
   INFO:root:rank=0 pagerank=6.6270e-01 url=4
   INFO:root:rank=1 pagerank=5.2179e-01 url=6
   INFO:root:rank=2 pagerank=4.1434e-01 url=5
   INFO:root:rank=3 pagerank=2.3175e-01 url=2
   INFO:root:rank=4 pagerank=1.8590e-01 url=3
   INFO:root:rank=5 pagerank=1.6917e-01 url=1
   ```

   Task 1, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
   INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
   INFO:root:rank=1 pagerank=8.9228e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
   INFO:root:rank=2 pagerank=7.0394e-04 url=www.lawfareblog.com/britains-coronavirus-response
   INFO:root:rank=3 pagerank=6.9157e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
   INFO:root:rank=4 pagerank=6.7045e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
   INFO:root:rank=5 pagerank=6.6260e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
   INFO:root:rank=6 pagerank=6.5050e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
   INFO:root:rank=7 pagerank=6.3623e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
   INFO:root:rank=8 pagerank=6.1252e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
   INFO:root:rank=9 pagerank=6.0191e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
   INFO:root:rank=0 pagerank=5.7827e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
   INFO:root:rank=1 pagerank=5.2340e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
   INFO:root:rank=2 pagerank=5.1298e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
   INFO:root:rank=3 pagerank=4.6601e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
   INFO:root:rank=4 pagerank=4.5935e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
   INFO:root:rank=5 pagerank=4.3073e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
   INFO:root:rank=6 pagerank=4.0936e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
   INFO:root:rank=7 pagerank=3.7592e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
   INFO:root:rank=8 pagerank=3.4510e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
   INFO:root:rank=9 pagerank=3.4486e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
   INFO:root:rank=0 pagerank=4.5747e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
   INFO:root:rank=1 pagerank=4.4175e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
   INFO:root:rank=2 pagerank=2.6929e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
   INFO:root:rank=3 pagerank=1.9392e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
   INFO:root:rank=4 pagerank=1.5453e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
   INFO:root:rank=5 pagerank=1.5358e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
   INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
   INFO:root:rank=7 pagerank=1.4222e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
   INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
   INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force

   ```

   Task 1, part 3:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz
   INFO:root:rank=0 pagerank=8.4165e+00 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
   INFO:root:rank=1 pagerank=8.4165e+00 url=www.lawfareblog.com/lawfare-job-board
   INFO:root:rank=2 pagerank=8.4165e+00 url=www.lawfareblog.com/documents-related-mueller-investigation
   INFO:root:rank=3 pagerank=8.4165e+00 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
   INFO:root:rank=4 pagerank=8.4165e+00 url=www.lawfareblog.com/subscribe-lawfare
   INFO:root:rank=5 pagerank=8.4165e+00 url=www.lawfareblog.com/masthead
   INFO:root:rank=6 pagerank=8.4165e+00 url=www.lawfareblog.com/topics
   INFO:root:rank=7 pagerank=8.4165e+00 url=www.lawfareblog.com/our-comments-policy
   INFO:root:rank=8 pagerank=8.4165e+00 url=www.lawfareblog.com/upcoming-events
   INFO:root:rank=9 pagerank=8.4165e+00 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site


   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
   INFO:root:rank=0 pagerank=3.4697e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
   INFO:root:rank=1 pagerank=2.9522e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
   INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
   INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
   INFO:root:rank=4 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
   INFO:root:rank=5 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
   INFO:root:rank=6 pagerank=1.5072e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
   INFO:root:rank=7 pagerank=1.4958e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
   INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
   INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
   ```

   Task 1, part 4:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
   DEBUG:root:computing indices
   DEBUG:root:computing values
   DEBUG:root:i=0 residual=141.904296875
   DEBUG:root:i=1 residual=0.11642682552337646
   DEBUG:root:i=2 residual=0.07496178895235062
   DEBUG:root:i=3 residual=0.03170211240649223
   DEBUG:root:i=4 residual=0.017446596175432205
   DEBUG:root:i=5 residual=0.00852623675018549
   DEBUG:root:i=6 residual=0.004441831726580858
   DEBUG:root:i=7 residual=0.002243307651951909
   DEBUG:root:i=8 residual=0.0011496387887746096
   DEBUG:root:i=9 residual=0.000581175263505429
   DEBUG:root:i=10 residual=0.0002926718443632126
   DEBUG:root:i=11 residual=0.00014554236258845776
   DEBUG:root:i=12 residual=7.149996235966682e-05
   DEBUG:root:i=13 residual=3.474645200185478e-05
   DEBUG:root:i=14 residual=1.595494723005686e-05
   DEBUG:root:i=15 residual=6.451506123994477e-06
   DEBUG:root:i=16 residual=2.4453806872770656e-06
   DEBUG:root:i=17 residual=8.108945621643215e-07
   INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
   INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
   INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
   INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
   INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/subscribe-lawfare
   INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
   INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
   INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
   INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
   INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/topics

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
   DEBUG:root:computing indices
   DEBUG:root:computing values
   DEBUG:root:i=0 residual=141.91505432128906
   DEBUG:root:i=1 residual=0.07088145613670349
   DEBUG:root:i=2 residual=0.01882273517549038
   DEBUG:root:i=3 residual=0.006958307698369026
   DEBUG:root:i=4 residual=0.002735827350988984
   DEBUG:root:i=5 residual=0.0010345602640882134
   DEBUG:root:i=6 residual=0.00037746390444226563
   DEBUG:root:i=7 residual=0.0001353339321212843
   DEBUG:root:i=8 residual=4.8224548663711175e-05
   DEBUG:root:i=9 residual=1.7172778825624846e-05
   DEBUG:root:i=10 residual=6.1153864407970104e-06
   DEBUG:root:i=11 residual=2.1725597889599157e-06
   DEBUG:root:i=12 residual=7.759175559840514e-07
   INFO:root:rank=0 pagerank=2.8859e-01 url=www.lawfareblog.com/snowden-revelations
   INFO:root:rank=1 pagerank=2.8859e-01 url=www.lawfareblog.com/lawfare-job-board
   INFO:root:rank=2 pagerank=2.8859e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
   INFO:root:rank=3 pagerank=2.8859e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
   INFO:root:rank=4 pagerank=2.8859e-01 url=www.lawfareblog.com/subscribe-lawfare
   INFO:root:rank=5 pagerank=2.8859e-01 url=www.lawfareblog.com/topics
   INFO:root:rank=6 pagerank=2.8859e-01 url=www.lawfareblog.com/masthead
   INFO:root:rank=7 pagerank=2.8859e-01 url=www.lawfareblog.com/our-comments-policy
   INFO:root:rank=8 pagerank=2.8859e-01 url=www.lawfareblog.com/upcoming-events
   INFO:root:rank=9 pagerank=2.8859e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
   DEBUG:root:computing indices
   DEBUG:root:computing values
   DEBUG:root:i=0 residual=133.37039184570312
   DEBUG:root:i=1 residual=0.49857109785079956
   DEBUG:root:i=2 residual=0.13418611884117126
   DEBUG:root:i=3 residual=0.06922280788421631
   DEBUG:root:i=4 residual=0.023409772664308548
   DEBUG:root:i=5 residual=0.010187196545302868
   DEBUG:root:i=6 residual=0.004906984977424145
   DEBUG:root:i=7 residual=0.002280223648995161
   DEBUG:root:i=8 residual=0.0010745085310190916
   DEBUG:root:i=9 residual=0.0005251344991847873
   DEBUG:root:i=10 residual=0.00026975717628374696
   DEBUG:root:i=11 residual=0.0001456816535210237
   DEBUG:root:i=12 residual=8.226733916671947e-05
   DEBUG:root:i=13 residual=4.8137273552129045e-05
   DEBUG:root:i=14 residual=2.880072861444205e-05
   DEBUG:root:i=15 residual=1.7428821593057364e-05
   DEBUG:root:i=16 residual=1.053675623552408e-05
   DEBUG:root:i=17 residual=6.389944246620871e-06
   DEBUG:root:i=18 residual=3.85494149668375e-06
   DEBUG:root:i=19 residual=2.2947599518374773e-06
   DEBUG:root:i=20 residual=1.3701263696930255e-06
   DEBUG:root:i=21 residual=8.087206992968277e-07
   INFO:root:rank=0 pagerank=3.4697e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
   INFO:root:rank=1 pagerank=2.9522e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
   INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
   INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
   INFO:root:rank=4 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
   INFO:root:rank=5 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
   INFO:root:rank=6 pagerank=1.5072e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
   INFO:root:rank=7 pagerank=1.4958e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
   INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
   INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull    

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
   DEBUG:root:computing indices
   DEBUG:root:computing values
   DEBUG:root:i=0 residual=133.93057250976562
   DEBUG:root:i=1 residual=0.5695649981498718
   DEBUG:root:i=2 residual=0.38299471139907837
   DEBUG:root:i=3 residual=0.21739359200000763
   DEBUG:root:i=4 residual=0.140450581908226
   DEBUG:root:i=5 residual=0.10851352661848068
   DEBUG:root:i=6 residual=0.09284140914678574
   DEBUG:root:i=7 residual=0.082255519926548
   DEBUG:root:i=8 residual=0.07338888943195343
   DEBUG:root:i=9 residual=0.06561233848333359
   DEBUG:root:i=10 residual=0.05909651517868042
   DEBUG:root:i=11 residual=0.05417541041970253
   DEBUG:root:i=12 residual=0.05111696198582649
   DEBUG:root:i=13 residual=0.04999380186200142
   DEBUG:root:i=14 residual=0.05060895159840584
   DEBUG:root:i=15 residual=0.05252622067928314
   DEBUG:root:i=16 residual=0.055188778787851334
   DEBUG:root:i=17 residual=0.05803852528333664
   DEBUG:root:i=18 residual=0.060592375695705414
   DEBUG:root:i=19 residual=0.062478456646203995
   DEBUG:root:i=20 residual=0.06345321983098984
   DEBUG:root:i=21 residual=0.06340522319078445
   DEBUG:root:i=22 residual=0.06234564259648323
   DEBUG:root:i=23 residual=0.060383718460798264
   DEBUG:root:i=24 residual=0.057693906128406525
   DEBUG:root:i=25 residual=0.05447975546121597
   DEBUG:root:i=26 residual=0.05094273388385773
   DEBUG:root:i=27 residual=0.04726110398769379
   DEBUG:root:i=28 residual=0.0435786247253418
   DEBUG:root:i=29 residual=0.040001653134822845
   DEBUG:root:i=30 residual=0.03660230711102486
   DEBUG:root:i=31 residual=0.03342437744140625
   DEBUG:root:i=32 residual=0.03048934042453766
   DEBUG:root:i=33 residual=0.027803216129541397
   DEBUG:root:i=34 residual=0.025360675528645515
   DEBUG:root:i=35 residual=0.023149995133280754
   DEBUG:root:i=36 residual=0.021155323833227158
   DEBUG:root:i=37 residual=0.019359201192855835
   DEBUG:root:i=38 residual=0.017743580043315887
   DEBUG:root:i=39 residual=0.01629081554710865
   DEBUG:root:i=40 residual=0.014984305016696453
   DEBUG:root:i=41 residual=0.013808632269501686
   DEBUG:root:i=42 residual=0.01274976134300232
   DEBUG:root:i=43 residual=0.011794946156442165
   DEBUG:root:i=44 residual=0.010932876728475094
   DEBUG:root:i=45 residual=0.010153379291296005
   DEBUG:root:i=46 residual=0.009447422809898853
   DEBUG:root:i=47 residual=0.008807025849819183
   DEBUG:root:i=48 residual=0.00822512898594141
   DEBUG:root:i=49 residual=0.007695477455854416
   DEBUG:root:i=50 residual=0.007212574128061533
   DEBUG:root:i=51 residual=0.006771526299417019
   DEBUG:root:i=52 residual=0.006367964204400778
   DEBUG:root:i=53 residual=0.005998102482408285
   DEBUG:root:i=54 residual=0.005658593028783798
   DEBUG:root:i=55 residual=0.005346349906176329
   DEBUG:root:i=56 residual=0.005058769602328539
   DEBUG:root:i=57 residual=0.004793411120772362
   DEBUG:root:i=58 residual=0.004548231139779091
   DEBUG:root:i=59 residual=0.004321304615586996
   DEBUG:root:i=60 residual=0.004110954236239195
   DEBUG:root:i=61 residual=0.003915725275874138
   DEBUG:root:i=62 residual=0.0037341900169849396
   DEBUG:root:i=63 residual=0.003565186634659767
   DEBUG:root:i=64 residual=0.003407623153179884
   DEBUG:root:i=65 residual=0.0032605458982288837
   DEBUG:root:i=66 residual=0.003123005386441946
   DEBUG:root:i=67 residual=0.0029942744877189398
   DEBUG:root:i=68 residual=0.0028736242093145847
   DEBUG:root:i=69 residual=0.0027603860944509506
   DEBUG:root:i=70 residual=0.0026539883110672235
   DEBUG:root:i=71 residual=0.002553920727223158
   DEBUG:root:i=72 residual=0.0024596485309302807
   DEBUG:root:i=73 residual=0.002370774280279875
   DEBUG:root:i=74 residual=0.002286874921992421
   DEBUG:root:i=75 residual=0.0022076196037232876
   DEBUG:root:i=76 residual=0.002132631605491042
   DEBUG:root:i=77 residual=0.0020616399124264717
   DEBUG:root:i=78 residual=0.0019943425431847572
   DEBUG:root:i=79 residual=0.001930505270138383
   DEBUG:root:i=80 residual=0.001869864296168089
   DEBUG:root:i=81 residual=0.0018122503533959389
   DEBUG:root:i=82 residual=0.0017574134981259704
   DEBUG:root:i=83 residual=0.0017052061157301068
   DEBUG:root:i=84 residual=0.0016554557951167226
   DEBUG:root:i=85 residual=0.0016080166678875685
   DEBUG:root:i=86 residual=0.0015627082902938128
   DEBUG:root:i=87 residual=0.0015194341540336609
   DEBUG:root:i=88 residual=0.0014780489727854729
   DEBUG:root:i=89 residual=0.0014384721871465445
   DEBUG:root:i=90 residual=0.0014005490811541677
   DEBUG:root:i=91 residual=0.0013642224948853254
   DEBUG:root:i=92 residual=0.0013293855590745807
   DEBUG:root:i=93 residual=0.0012959609739482403
   DEBUG:root:i=94 residual=0.0012638624757528305
   DEBUG:root:i=95 residual=0.0012330381432548165
   DEBUG:root:i=96 residual=0.0012033922830596566
   DEBUG:root:i=97 residual=0.0011748677352443337
   DEBUG:root:i=98 residual=0.0011474067578092217
   DEBUG:root:i=99 residual=0.0011209642980247736
   DEBUG:root:i=100 residual=0.001095483428798616
   DEBUG:root:i=101 residual=0.001070912228897214
   DEBUG:root:i=102 residual=0.0010471923742443323
   DEBUG:root:i=103 residual=0.0010242972057312727
   DEBUG:root:i=104 residual=0.001002186443656683
   DEBUG:root:i=105 residual=0.0009808100294321775
   DEBUG:root:i=106 residual=0.0009601528872735798
   DEBUG:root:i=107 residual=0.0009401642601005733
   DEBUG:root:i=108 residual=0.000920831342227757
   DEBUG:root:i=109 residual=0.0009020977304317057
   DEBUG:root:i=110 residual=0.0008839466026984155
   DEBUG:root:i=111 residual=0.0008663698099553585
   DEBUG:root:i=112 residual=0.0008493220084346831
   DEBUG:root:i=113 residual=0.000832771765999496
   DEBUG:root:i=114 residual=0.0008167207706719637
   DEBUG:root:i=115 residual=0.0008011477766558528
   DEBUG:root:i=116 residual=0.0007860091864131391
   DEBUG:root:i=117 residual=0.0007713031955063343
   DEBUG:root:i=118 residual=0.0007570118177682161
   DEBUG:root:i=119 residual=0.0007431185804307461
   DEBUG:root:i=120 residual=0.0007295927498489618
   DEBUG:root:i=121 residual=0.0007164450362324715
   DEBUG:root:i=122 residual=0.0007036390597932041
   DEBUG:root:i=123 residual=0.0006911737145856023
   DEBUG:root:i=124 residual=0.000679036311339587
   DEBUG:root:i=125 residual=0.0006672025192528963
   DEBUG:root:i=126 residual=0.0006556721054948866
   DEBUG:root:i=127 residual=0.0006444292957894504
   DEBUG:root:i=128 residual=0.0006334552890621126
   DEBUG:root:i=129 residual=0.000622765626758337
   DEBUG:root:i=130 residual=0.0006123253842815757
   DEBUG:root:i=131 residual=0.0006021384033374488
   DEBUG:root:i=132 residual=0.0005921872216276824
   DEBUG:root:i=133 residual=0.0005824664840474725
   DEBUG:root:i=134 residual=0.0005729847471229732
   DEBUG:root:i=135 residual=0.0005637064459733665
   DEBUG:root:i=136 residual=0.000554638565517962
   DEBUG:root:i=137 residual=0.000545780174434185
   DEBUG:root:i=138 residual=0.0005371168372221291
   DEBUG:root:i=139 residual=0.0005286384839564562
   DEBUG:root:i=140 residual=0.0005203488399274647
   DEBUG:root:i=141 residual=0.0005122339352965355
   DEBUG:root:i=142 residual=0.0005042958655394614
   DEBUG:root:i=143 residual=0.0004965217667631805
   DEBUG:root:i=144 residual=0.0004889040137641132
   DEBUG:root:i=145 residual=0.0004814552958123386
   DEBUG:root:i=146 residual=0.0004741549491882324
   DEBUG:root:i=147 residual=0.00046699901577085257
   DEBUG:root:i=148 residual=0.00045998883433640003
   DEBUG:root:i=149 residual=0.00045311899157240987
   DEBUG:root:i=150 residual=0.000446390884462744
   DEBUG:root:i=151 residual=0.000439786643255502
   DEBUG:root:i=152 residual=0.00043331680353730917
   DEBUG:root:i=153 residual=0.0004269657365512103
   DEBUG:root:i=154 residual=0.0004207411257084459
   DEBUG:root:i=155 residual=0.00041462609078735113
   DEBUG:root:i=156 residual=0.00040863605681806803
   DEBUG:root:i=157 residual=0.0004027567047160119
   DEBUG:root:i=158 residual=0.0003969827084802091
   DEBUG:root:i=159 residual=0.00039131706580519676
   DEBUG:root:i=160 residual=0.000385751249268651
   DEBUG:root:i=161 residual=0.00038029070128686726
   DEBUG:root:i=162 residual=0.0003749222378246486
   DEBUG:root:i=163 residual=0.0003696540661621839
   DEBUG:root:i=164 residual=0.0003644807147793472
   DEBUG:root:i=165 residual=0.00035939368535764515
   DEBUG:root:i=166 residual=0.0003543985658325255
   DEBUG:root:i=167 residual=0.00034948784741573036
   DEBUG:root:i=168 residual=0.0003446677001193166
   DEBUG:root:i=169 residual=0.00033992587123066187
   DEBUG:root:i=170 residual=0.000335266871843487
   DEBUG:root:i=171 residual=0.0003306878497824073
   DEBUG:root:i=172 residual=0.0003261821111664176
   DEBUG:root:i=173 residual=0.00032175707747228444
   DEBUG:root:i=174 residual=0.00031739965197630227
   DEBUG:root:i=175 residual=0.00031312197097577155
   DEBUG:root:i=176 residual=0.0003089102974627167
   DEBUG:root:i=177 residual=0.00030476771644316614
   DEBUG:root:i=178 residual=0.0003006898914463818
   DEBUG:root:i=179 residual=0.00029668258503079414
   DEBUG:root:i=180 residual=0.0002927393652498722
   DEBUG:root:i=181 residual=0.0002888553135562688
   DEBUG:root:i=182 residual=0.00028503674548119307
   DEBUG:root:i=183 residual=0.000281275890301913
   DEBUG:root:i=184 residual=0.00027757693897001445
   DEBUG:root:i=185 residual=0.00027393567143008113
   DEBUG:root:i=186 residual=0.00027034952654503286
   DEBUG:root:i=187 residual=0.0002668235683813691
   DEBUG:root:i=188 residual=0.0002633505209814757
   DEBUG:root:i=189 residual=0.00025992849259637296
   DEBUG:root:i=190 residual=0.00025655797799117863
   DEBUG:root:i=191 residual=0.0002532429061830044
   DEBUG:root:i=192 residual=0.00024997416767291725
   DEBUG:root:i=193 residual=0.0002467572339810431
   DEBUG:root:i=194 residual=0.0002435892674839124
   DEBUG:root:i=195 residual=0.00024046770704444498
   DEBUG:root:i=196 residual=0.00023739278549328446
   DEBUG:root:i=197 residual=0.00023436346964444965
   DEBUG:root:i=198 residual=0.00023138141841627657
   DEBUG:root:i=199 residual=0.00022844101476948708
   DEBUG:root:i=200 residual=0.00022554535826202482
   DEBUG:root:i=201 residual=0.00022269213513936847
   DEBUG:root:i=202 residual=0.00021988118533045053
   DEBUG:root:i=203 residual=0.00021710649889428169
   DEBUG:root:i=204 residual=0.00021437335817608982
   DEBUG:root:i=205 residual=0.00021168429520912468
   DEBUG:root:i=206 residual=0.0002090325579047203
   DEBUG:root:i=207 residual=0.0002064158907160163
   DEBUG:root:i=208 residual=0.0002038390375673771
   DEBUG:root:i=209 residual=0.00020129821496084332
   DEBUG:root:i=210 residual=0.00019879396131727844
   DEBUG:root:i=211 residual=0.0001963245013030246
   DEBUG:root:i=212 residual=0.0001938904169946909
   DEBUG:root:i=213 residual=0.00019148614956066012
   DEBUG:root:i=214 residual=0.00018912127416115254
   DEBUG:root:i=215 residual=0.00018678860215004534
   DEBUG:root:i=216 residual=0.00018448670743964612
   DEBUG:root:i=217 residual=0.00018221752543468028
   DEBUG:root:i=218 residual=0.00017998006660491228
   DEBUG:root:i=219 residual=0.00017777160974219441
   DEBUG:root:i=220 residual=0.00017559733532834798
   DEBUG:root:i=221 residual=0.00017345213564112782
   DEBUG:root:i=222 residual=0.00017133411893155426
   DEBUG:root:i=223 residual=0.00016924392548389733
   DEBUG:root:i=224 residual=0.00016718653205316514
   DEBUG:root:i=225 residual=0.00016515272727701813
   DEBUG:root:i=226 residual=0.00016314951062668115
   DEBUG:root:i=227 residual=0.00016117081395350397
   DEBUG:root:i=228 residual=0.00015921918384265155
   DEBUG:root:i=229 residual=0.00015729635197203606
   DEBUG:root:i=230 residual=0.00015539470768999308
   DEBUG:root:i=231 residual=0.000153523898916319
   DEBUG:root:i=232 residual=0.0001516752236057073
   DEBUG:root:i=233 residual=0.00014985157758928835
   DEBUG:root:i=234 residual=0.00014805061800871044
   DEBUG:root:i=235 residual=0.00014627478958573192
   DEBUG:root:i=236 residual=0.0001445223024347797
   DEBUG:root:i=237 residual=0.00014279426250141114
   DEBUG:root:i=238 residual=0.00014108569303061813
   DEBUG:root:i=239 residual=0.00013940205099061131
   DEBUG:root:i=240 residual=0.00013774065882898867
   DEBUG:root:i=241 residual=0.00013609827146865427
   DEBUG:root:i=242 residual=0.00013447983656078577
   DEBUG:root:i=243 residual=0.00013288352056406438
   DEBUG:root:i=244 residual=0.00013130405568517745
   DEBUG:root:i=245 residual=0.0001297475246246904
   DEBUG:root:i=246 residual=0.00012820929987356067
   DEBUG:root:i=247 residual=0.00012669268471654505
   DEBUG:root:i=248 residual=0.0001251936482731253
   DEBUG:root:i=249 residual=0.00012371820048429072
   DEBUG:root:i=250 residual=0.0001222574064740911
   DEBUG:root:i=251 residual=0.00012081601744284853
   DEBUG:root:i=252 residual=0.00011939669639104977
   DEBUG:root:i=253 residual=0.00011799136700574309
   DEBUG:root:i=254 residual=0.00011660598829621449
   DEBUG:root:i=255 residual=0.00011523488501552492
   DEBUG:root:i=256 residual=0.00011388439452275634
   DEBUG:root:i=257 residual=0.0001125502894865349
   DEBUG:root:i=258 residual=0.00011123131116619334
   DEBUG:root:i=259 residual=0.00010993080650223419
   DEBUG:root:i=260 residual=0.00010864744399441406
   DEBUG:root:i=261 residual=0.00010737866250565276
   DEBUG:root:i=262 residual=0.00010612511687213555
   DEBUG:root:i=263 residual=0.00010488672705832869
   DEBUG:root:i=264 residual=0.0001036647372529842
   DEBUG:root:i=265 residual=0.00010245986049994826
   DEBUG:root:i=266 residual=0.00010126688721356913
   DEBUG:root:i=267 residual=0.00010009081597672775
   DEBUG:root:i=268 residual=9.892856905935332e-05
   DEBUG:root:i=269 residual=9.778088133316487e-05
   DEBUG:root:i=270 residual=9.664880781201646e-05
   DEBUG:root:i=271 residual=9.552962001180276e-05
   DEBUG:root:i=272 residual=9.442531882086769e-05
   DEBUG:root:i=273 residual=9.333391062682495e-05
   DEBUG:root:i=274 residual=9.225465328199789e-05
   DEBUG:root:i=275 residual=9.118948219111189e-05
   DEBUG:root:i=276 residual=9.013723320094869e-05
   DEBUG:root:i=277 residual=8.910108590498567e-05
   DEBUG:root:i=278 residual=8.80728111951612e-05
   DEBUG:root:i=279 residual=8.705967775313184e-05
   DEBUG:root:i=280 residual=8.605796028859913e-05
   DEBUG:root:i=281 residual=8.506979793310165e-05
   DEBUG:root:i=282 residual=8.409240399487317e-05
   DEBUG:root:i=283 residual=8.31283614388667e-05
   DEBUG:root:i=284 residual=8.217316644731909e-05
   DEBUG:root:i=285 residual=8.123328007059172e-05
   DEBUG:root:i=286 residual=8.030123717617244e-05
   DEBUG:root:i=287 residual=7.938318594824523e-05
   DEBUG:root:i=288 residual=7.847433153074235e-05
   DEBUG:root:i=289 residual=7.757703133393079e-05
   DEBUG:root:i=290 residual=7.669032493140548e-05
   DEBUG:root:i=291 residual=7.581352110719308e-05
   DEBUG:root:i=292 residual=7.495110185118392e-05
   DEBUG:root:i=293 residual=7.409483805531636e-05
   DEBUG:root:i=294 residual=7.325079059228301e-05
   DEBUG:root:i=295 residual=7.24146084394306e-05
   DEBUG:root:i=296 residual=7.159137021517381e-05
   DEBUG:root:i=297 residual=7.077698683133349e-05
   DEBUG:root:i=298 residual=6.997198215685785e-05
   DEBUG:root:i=299 residual=6.917848077137023e-05
   DEBUG:root:i=300 residual=6.839162233518437e-05
   DEBUG:root:i=301 residual=6.761457916582003e-05
   DEBUG:root:i=302 residual=6.684701656922698e-05
   DEBUG:root:i=303 residual=6.60886216792278e-05
   DEBUG:root:i=304 residual=6.53417082503438e-05
   DEBUG:root:i=305 residual=6.460022268583998e-05
   DEBUG:root:i=306 residual=6.386918539647013e-05
   DEBUG:root:i=307 residual=6.314593338174745e-05
   DEBUG:root:i=308 residual=6.243272946448997e-05
   DEBUG:root:i=309 residual=6.172636494738981e-05
   DEBUG:root:i=310 residual=6.1028506024740636e-05
   DEBUG:root:i=311 residual=6.033801400917582e-05
   DEBUG:root:i=312 residual=5.9657872043317184e-05
   DEBUG:root:i=313 residual=5.8986137446481735e-05
   DEBUG:root:i=314 residual=5.831873568240553e-05
   DEBUG:root:i=315 residual=5.7661469327285886e-05
   DEBUG:root:i=316 residual=5.7009201555047184e-05
   DEBUG:root:i=317 residual=5.636891000904143e-05
   DEBUG:root:i=318 residual=5.5733631597831845e-05
   DEBUG:root:i=319 residual=5.510662595042959e-05
   DEBUG:root:i=320 residual=5.448685260489583e-05
   DEBUG:root:i=321 residual=5.3873518481850624e-05
   DEBUG:root:i=322 residual=5.3268206102075055e-05
   DEBUG:root:i=323 residual=5.267083179205656e-05
   DEBUG:root:i=324 residual=5.2078736189287156e-05
   DEBUG:root:i=325 residual=5.149419666850008e-05
   DEBUG:root:i=326 residual=5.091695857117884e-05
   DEBUG:root:i=327 residual=5.03444935020525e-05
   DEBUG:root:i=328 residual=4.978167271474376e-05
   DEBUG:root:i=329 residual=4.922291191178374e-05
   DEBUG:root:i=330 residual=4.8669506213627756e-05
   DEBUG:root:i=331 residual=4.812582483282313e-05
   DEBUG:root:i=332 residual=4.7586963773937896e-05
   DEBUG:root:i=333 residual=4.705562241724692e-05
   DEBUG:root:i=334 residual=4.6527762606274337e-05
   DEBUG:root:i=335 residual=4.60075716546271e-05
   DEBUG:root:i=336 residual=4.549308505374938e-05
   DEBUG:root:i=337 residual=4.498465932556428e-05
   DEBUG:root:i=338 residual=4.448326217243448e-05
   DEBUG:root:i=339 residual=4.398645251058042e-05
   DEBUG:root:i=340 residual=4.349557275418192e-05
   DEBUG:root:i=341 residual=4.3009691580664366e-05
   DEBUG:root:i=342 residual=4.253204679116607e-05
   DEBUG:root:i=343 residual=4.2054292862303555e-05
   DEBUG:root:i=344 residual=4.158791125519201e-05
   DEBUG:root:i=345 residual=4.1124600102193654e-05
   DEBUG:root:i=346 residual=4.066635301569477e-05
   DEBUG:root:i=347 residual=4.02145778934937e-05
   DEBUG:root:i=348 residual=3.976582956966013e-05
   DEBUG:root:i=349 residual=3.932368417736143e-05
   DEBUG:root:i=350 residual=3.888700302923098e-05
   DEBUG:root:i=351 residual=3.845342871500179e-05
   DEBUG:root:i=352 residual=3.8027854316169396e-05
   DEBUG:root:i=353 residual=3.760385880013928e-05
   DEBUG:root:i=354 residual=3.7186400732025504e-05
   DEBUG:root:i=355 residual=3.677211134345271e-05
   DEBUG:root:i=356 residual=3.63641265721526e-05
   DEBUG:root:i=357 residual=3.596072565414943e-05
   DEBUG:root:i=358 residual=3.556230149115436e-05
   DEBUG:root:i=359 residual=3.516681681503542e-05
   DEBUG:root:i=360 residual=3.47785244230181e-05
   DEBUG:root:i=361 residual=3.4392389352433383e-05
   DEBUG:root:i=362 residual=3.401066715014167e-05
   DEBUG:root:i=363 residual=3.363318683113903e-05
   DEBUG:root:i=364 residual=3.326210207887925e-05
   DEBUG:root:i=365 residual=3.289267988293432e-05
   DEBUG:root:i=366 residual=3.25295768561773e-05
   DEBUG:root:i=367 residual=3.216825280105695e-05
   DEBUG:root:i=368 residual=3.1812440283829346e-05
   DEBUG:root:i=369 residual=3.146063318126835e-05
   DEBUG:root:i=370 residual=3.11143921862822e-05
   DEBUG:root:i=371 residual=3.0770203011343256e-05
   DEBUG:root:i=372 residual=3.0429115213337354e-05
   DEBUG:root:i=373 residual=3.0092531233094633e-05
   DEBUG:root:i=374 residual=2.9760572942905128e-05
   DEBUG:root:i=375 residual=2.9430817448883317e-05
   DEBUG:root:i=376 residual=2.910527473432012e-05
   DEBUG:root:i=377 residual=2.8784832466044463e-05
   DEBUG:root:i=378 residual=2.8468206437537447e-05
   DEBUG:root:i=379 residual=2.815352490870282e-05
   DEBUG:root:i=380 residual=2.7843285351991653e-05
   DEBUG:root:i=381 residual=2.753507396846544e-05
   DEBUG:root:i=382 residual=2.7230344130657613e-05
   DEBUG:root:i=383 residual=2.6931989850709215e-05
   DEBUG:root:i=384 residual=2.663413033587858e-05
   DEBUG:root:i=385 residual=2.6341554985265248e-05
   DEBUG:root:i=386 residual=2.6050158339785412e-05
   DEBUG:root:i=387 residual=2.576439146650955e-05
   DEBUG:root:i=388 residual=2.5480161639279686e-05
   DEBUG:root:i=389 residual=2.520013731555082e-05
   DEBUG:root:i=390 residual=2.4921248041209765e-05
   DEBUG:root:i=391 residual=2.4648104954394512e-05
   DEBUG:root:i=392 residual=2.437762486806605e-05
   DEBUG:root:i=393 residual=2.4107815988827497e-05
   DEBUG:root:i=394 residual=2.3843467715778388e-05
   DEBUG:root:i=395 residual=2.3581107598147355e-05
   DEBUG:root:i=396 residual=2.3321699700318277e-05
   DEBUG:root:i=397 residual=2.30647037824383e-05
   DEBUG:root:i=398 residual=2.2812519091530703e-05
   DEBUG:root:i=399 residual=2.2560261641046964e-05
   DEBUG:root:i=400 residual=2.2312251530820504e-05
   DEBUG:root:i=401 residual=2.2068197722546756e-05
   DEBUG:root:i=402 residual=2.182471871492453e-05
   DEBUG:root:i=403 residual=2.1583904526778497e-05
   DEBUG:root:i=404 residual=2.1347777874325402e-05
   DEBUG:root:i=405 residual=2.1114306946401484e-05
   DEBUG:root:i=406 residual=2.088129440380726e-05
   DEBUG:root:i=407 residual=2.065391709038522e-05
   DEBUG:root:i=408 residual=2.0426628907443956e-05
   DEBUG:root:i=409 residual=2.0202189261908643e-05
   DEBUG:root:i=410 residual=1.998096013267059e-05
   DEBUG:root:i=411 residual=1.9760866052820347e-05
   DEBUG:root:i=412 residual=1.9544622773537412e-05
   DEBUG:root:i=413 residual=1.9330154827912338e-05
   DEBUG:root:i=414 residual=1.9118944692309014e-05
   DEBUG:root:i=415 residual=1.8908325728261843e-05
   DEBUG:root:i=416 residual=1.8700995497056283e-05
   DEBUG:root:i=417 residual=1.8496031771064736e-05
   DEBUG:root:i=418 residual=1.8294236724614166e-05
   DEBUG:root:i=419 residual=1.8093991457135417e-05
   DEBUG:root:i=420 residual=1.7896341887535527e-05
   DEBUG:root:i=421 residual=1.7699117961456068e-05
   DEBUG:root:i=422 residual=1.7505415598861873e-05
   DEBUG:root:i=423 residual=1.731268821458798e-05
   DEBUG:root:i=424 residual=1.712457014946267e-05
   DEBUG:root:i=425 residual=1.6935788153205067e-05
   DEBUG:root:i=426 residual=1.67517446243437e-05
   DEBUG:root:i=427 residual=1.656983295106329e-05
   DEBUG:root:i=428 residual=1.638828507566359e-05
   DEBUG:root:i=429 residual=1.6207459339057095e-05
   DEBUG:root:i=430 residual=1.6030482584028505e-05
   DEBUG:root:i=431 residual=1.5855641322559677e-05
   DEBUG:root:i=432 residual=1.5682097000535578e-05
   DEBUG:root:i=433 residual=1.550982051412575e-05
   DEBUG:root:i=434 residual=1.5340881873271428e-05
   DEBUG:root:i=435 residual=1.5173450265137944e-05
   DEBUG:root:i=436 residual=1.5006856301624794e-05
   DEBUG:root:i=437 residual=1.4844822544546332e-05
   DEBUG:root:i=438 residual=1.468129539716756e-05
   DEBUG:root:i=439 residual=1.4520168406306766e-05
   DEBUG:root:i=440 residual=1.4362604815687519e-05
   DEBUG:root:i=441 residual=1.4206156265572645e-05
   DEBUG:root:i=442 residual=1.4049251149117481e-05
   DEBUG:root:i=443 residual=1.3894118637836073e-05
   DEBUG:root:i=444 residual=1.374478597426787e-05
   DEBUG:root:i=445 residual=1.3592491086455993e-05
   DEBUG:root:i=446 residual=1.3445431250147521e-05
   DEBUG:root:i=447 residual=1.3298632438818458e-05
   DEBUG:root:i=448 residual=1.3154228327039164e-05
   DEBUG:root:i=449 residual=1.3010913789912593e-05
   DEBUG:root:i=450 residual=1.2868047633673996e-05
   DEBUG:root:i=451 residual=1.27261055240524e-05
   DEBUG:root:i=452 residual=1.2588633580890018e-05
   DEBUG:root:i=453 residual=1.2449748282961082e-05
   DEBUG:root:i=454 residual=1.2315866115386598e-05
   DEBUG:root:i=455 residual=1.2182673344796058e-05
   DEBUG:root:i=456 residual=1.2047889867972117e-05
   DEBUG:root:i=457 residual=1.1916705261683092e-05
   DEBUG:root:i=458 residual=1.1785027709265705e-05
   DEBUG:root:i=459 residual=1.1656688911898527e-05
   DEBUG:root:i=460 residual=1.153023822553223e-05
   DEBUG:root:i=461 residual=1.1404159522498958e-05
   DEBUG:root:i=462 residual=1.1280200851615518e-05
   DEBUG:root:i=463 residual=1.1156753316754475e-05
   DEBUG:root:i=464 residual=1.1035702300432604e-05
   DEBUG:root:i=465 residual=1.0915734492300544e-05
   DEBUG:root:i=466 residual=1.0797174581966829e-05
   DEBUG:root:i=467 residual=1.0679083061404526e-05
   DEBUG:root:i=468 residual=1.0562162060523406e-05
   DEBUG:root:i=469 residual=1.0446991836943198e-05
   DEBUG:root:i=470 residual=1.0333833415643312e-05
   DEBUG:root:i=471 residual=1.0221629963780288e-05
   DEBUG:root:i=472 residual=1.0110072253155522e-05
   DEBUG:root:i=473 residual=9.998748282669112e-06
   DEBUG:root:i=474 residual=9.889703505905345e-06
   DEBUG:root:i=475 residual=9.782036613614764e-06
   DEBUG:root:i=476 residual=9.674904504208826e-06
   DEBUG:root:i=477 residual=9.569968824507669e-06
   DEBUG:root:i=478 residual=9.466260962653905e-06
   DEBUG:root:i=479 residual=9.363147910335101e-06
   DEBUG:root:i=480 residual=9.260829756385647e-06
   DEBUG:root:i=481 residual=9.162211426883005e-06
   DEBUG:root:i=482 residual=9.061977834790014e-06
   DEBUG:root:i=483 residual=8.963849722931627e-06
   DEBUG:root:i=484 residual=8.865114068612456e-06
   DEBUG:root:i=485 residual=8.768890438659582e-06
   DEBUG:root:i=486 residual=8.674253876961302e-06
   DEBUG:root:i=487 residual=8.579080713388976e-06
   DEBUG:root:i=488 residual=8.484909812978003e-06
   DEBUG:root:i=489 residual=8.393443749810103e-06
   DEBUG:root:i=490 residual=8.302469723275863e-06
   DEBUG:root:i=491 residual=8.21206958789844e-06
   DEBUG:root:i=492 residual=8.122293365886435e-06
   DEBUG:root:i=493 residual=8.034711754589807e-06
   DEBUG:root:i=494 residual=7.946287041704636e-06
   DEBUG:root:i=495 residual=7.859805918997154e-06
   DEBUG:root:i=496 residual=7.774766345391981e-06
   DEBUG:root:i=497 residual=7.690034180996008e-06
   DEBUG:root:i=498 residual=7.606442977703409e-06
   DEBUG:root:i=499 residual=7.52539608583902e-06
   DEBUG:root:i=500 residual=7.441192792612128e-06
   DEBUG:root:i=501 residual=7.360977178905159e-06
   DEBUG:root:i=502 residual=7.281857961061178e-06
   DEBUG:root:i=503 residual=7.20218167771236e-06
   DEBUG:root:i=504 residual=7.1268409556068946e-06
   DEBUG:root:i=505 residual=7.0464948294102214e-06
   DEBUG:root:i=506 residual=6.9702614382549655e-06
   DEBUG:root:i=507 residual=6.896883405715926e-06
   DEBUG:root:i=508 residual=6.818905148975318e-06
   DEBUG:root:i=509 residual=6.744366601196816e-06
   DEBUG:root:i=510 residual=6.671372830169275e-06
   DEBUG:root:i=511 residual=6.59946954328916e-06
   DEBUG:root:i=512 residual=6.5293556872347835e-06
   DEBUG:root:i=513 residual=6.457156359829241e-06
   DEBUG:root:i=514 residual=6.38647588857566e-06
   DEBUG:root:i=515 residual=6.317893166851718e-06
   DEBUG:root:i=516 residual=6.249062607821543e-06
   DEBUG:root:i=517 residual=6.1811333580408245e-06
   DEBUG:root:i=518 residual=6.114173174864845e-06
   DEBUG:root:i=519 residual=6.048098384781042e-06
   DEBUG:root:i=520 residual=5.980976311548147e-06
   DEBUG:root:i=521 residual=5.9173512454435695e-06
   DEBUG:root:i=522 residual=5.853945367562119e-06
   DEBUG:root:i=523 residual=5.789594979432877e-06
   DEBUG:root:i=524 residual=5.726507879444398e-06
   DEBUG:root:i=525 residual=5.66398057344486e-06
   DEBUG:root:i=526 residual=5.603342742688255e-06
   DEBUG:root:i=527 residual=5.5423170124413446e-06
   DEBUG:root:i=528 residual=5.481655534822494e-06
   DEBUG:root:i=529 residual=5.422816684585996e-06
   DEBUG:root:i=530 residual=5.363411219150294e-06
   DEBUG:root:i=531 residual=5.3056587603350636e-06
   DEBUG:root:i=532 residual=5.247011358733289e-06
   DEBUG:root:i=533 residual=5.190160209167516e-06
   DEBUG:root:i=534 residual=5.13558552484028e-06
   DEBUG:root:i=535 residual=5.0781809477484785e-06
   DEBUG:root:i=536 residual=5.022326149628498e-06
   DEBUG:root:i=537 residual=4.969854217051761e-06
   DEBUG:root:i=538 residual=4.914005330647342e-06
   DEBUG:root:i=539 residual=4.86155113321729e-06
   DEBUG:root:i=540 residual=4.808460744243348e-06
   DEBUG:root:i=541 residual=4.756810994877014e-06
   DEBUG:root:i=542 residual=4.704951606981922e-06
   DEBUG:root:i=543 residual=4.653164978662971e-06
   DEBUG:root:i=544 residual=4.6043355723668355e-06
   DEBUG:root:i=545 residual=4.553970029519405e-06
   DEBUG:root:i=546 residual=4.504164735408267e-06
   DEBUG:root:i=547 residual=4.455773250811035e-06
   DEBUG:root:i=548 residual=4.4077587517676875e-06
   DEBUG:root:i=549 residual=4.3613290472421795e-06
   DEBUG:root:i=550 residual=4.3126055970788e-06
   DEBUG:root:i=551 residual=4.266317318979418e-06
   DEBUG:root:i=552 residual=4.219225502311019e-06
   DEBUG:root:i=553 residual=4.173272827756591e-06
   DEBUG:root:i=554 residual=4.128025921090739e-06
   DEBUG:root:i=555 residual=4.087171419087099e-06
   DEBUG:root:i=556 residual=4.038774932269007e-06
   DEBUG:root:i=557 residual=3.994266080553643e-06
   DEBUG:root:i=558 residual=3.9529427340312395e-06
   DEBUG:root:i=559 residual=3.908831786247902e-06
   DEBUG:root:i=560 residual=3.867626674036728e-06
   DEBUG:root:i=561 residual=3.8248167584242765e-06
   DEBUG:root:i=562 residual=3.78317668037198e-06
   DEBUG:root:i=563 residual=3.7411382436403073e-06
   DEBUG:root:i=564 residual=3.701543391798623e-06
   DEBUG:root:i=565 residual=3.6605918012355687e-06
   DEBUG:root:i=566 residual=3.621710902734776e-06
   DEBUG:root:i=567 residual=3.58353372575948e-06
   DEBUG:root:i=568 residual=3.546599373294157e-06
   DEBUG:root:i=569 residual=3.5059924812230747e-06
   DEBUG:root:i=570 residual=3.4675956612773007e-06
   DEBUG:root:i=571 residual=3.4307413443457335e-06
   DEBUG:root:i=572 residual=3.3932503811229253e-06
   DEBUG:root:i=573 residual=3.356291927048005e-06
   DEBUG:root:i=574 residual=3.3203016300831223e-06
   DEBUG:root:i=575 residual=3.283323621872114e-06
   DEBUG:root:i=576 residual=3.249812607464264e-06
   DEBUG:root:i=577 residual=3.2165983157028677e-06
   DEBUG:root:i=578 residual=3.177634880557889e-06
   DEBUG:root:i=579 residual=3.1444683372683357e-06
   DEBUG:root:i=580 residual=3.1089748517842963e-06
   DEBUG:root:i=581 residual=3.0758947104914114e-06
   DEBUG:root:i=582 residual=3.042826165255974e-06
   DEBUG:root:i=583 residual=3.0089022402535193e-06
   DEBUG:root:i=584 residual=2.977022404593299e-06
   DEBUG:root:i=585 residual=2.944518655567663e-06
   DEBUG:root:i=586 residual=2.9132445433788234e-06
   DEBUG:root:i=587 residual=2.88172464024683e-06
   DEBUG:root:i=588 residual=2.8513416054920526e-06
   DEBUG:root:i=589 residual=2.8205113267176785e-06
   DEBUG:root:i=590 residual=2.791935457935324e-06
   DEBUG:root:i=591 residual=2.7596772724791663e-06
   DEBUG:root:i=592 residual=2.7289056561130565e-06
   DEBUG:root:i=593 residual=2.703762220335193e-06
   DEBUG:root:i=594 residual=2.6686445835366612e-06
   DEBUG:root:i=595 residual=2.640415232235682e-06
   DEBUG:root:i=596 residual=2.614663117128657e-06
   DEBUG:root:i=597 residual=2.5837566681730095e-06
   DEBUG:root:i=598 residual=2.5573951916157966e-06
   DEBUG:root:i=599 residual=2.5295362320321146e-06
   DEBUG:root:i=600 residual=2.500776190572651e-06
   DEBUG:root:i=601 residual=2.4755177037150133e-06
   DEBUG:root:i=602 residual=2.4476792077621212e-06
   DEBUG:root:i=603 residual=2.4201165160775417e-06
   DEBUG:root:i=604 residual=2.3950071863509947e-06
   DEBUG:root:i=605 residual=2.36890628002584e-06
   DEBUG:root:i=606 residual=2.3487875751015963e-06
   DEBUG:root:i=607 residual=2.320077328477055e-06
   DEBUG:root:i=608 residual=2.2946026092540706e-06
   DEBUG:root:i=609 residual=2.2709102722728858e-06
   DEBUG:root:i=610 residual=2.244272764073685e-06
   DEBUG:root:i=611 residual=2.220437181676971e-06
   DEBUG:root:i=612 residual=2.1954749627184356e-06
   DEBUG:root:i=613 residual=2.171533424188965e-06
   DEBUG:root:i=614 residual=2.148265139112482e-06
   DEBUG:root:i=615 residual=2.1237851797195617e-06
   DEBUG:root:i=616 residual=2.1016455775679788e-06
   DEBUG:root:i=617 residual=2.078578518194263e-06
   DEBUG:root:i=618 residual=2.0558657070068875e-06
   DEBUG:root:i=619 residual=2.035279749179608e-06
   DEBUG:root:i=620 residual=2.0115840015932918e-06
   DEBUG:root:i=621 residual=1.9907301975763403e-06
   DEBUG:root:i=622 residual=1.9691119632625487e-06
   DEBUG:root:i=623 residual=1.9464705474092625e-06
   DEBUG:root:i=624 residual=1.9280832930235192e-06
   DEBUG:root:i=625 residual=1.906813281493669e-06
   DEBUG:root:i=626 residual=1.884004745988932e-06
   DEBUG:root:i=627 residual=1.8647473325472674e-06
   DEBUG:root:i=628 residual=1.8428935391057166e-06
   DEBUG:root:i=629 residual=1.8254756923852256e-06
   DEBUG:root:i=630 residual=1.8060693491861457e-06
   DEBUG:root:i=631 residual=1.786453026397794e-06
   DEBUG:root:i=632 residual=1.7661452602624195e-06
   DEBUG:root:i=633 residual=1.74687761500536e-06
   DEBUG:root:i=634 residual=1.7283878150919918e-06
   DEBUG:root:i=635 residual=1.7098209355026484e-06
   DEBUG:root:i=636 residual=1.6911325246837805e-06
   DEBUG:root:i=637 residual=1.6736828456487274e-06
   DEBUG:root:i=638 residual=1.6549822703382233e-06
   DEBUG:root:i=639 residual=1.6372745221815421e-06
   DEBUG:root:i=640 residual=1.6174315078387735e-06
   DEBUG:root:i=641 residual=1.6009454384402488e-06
   DEBUG:root:i=642 residual=1.5839863181099645e-06
   DEBUG:root:i=643 residual=1.5650116438337136e-06
   DEBUG:root:i=644 residual=1.549093553876446e-06
   DEBUG:root:i=645 residual=1.534390207780234e-06
   DEBUG:root:i=646 residual=1.516528868705791e-06
   DEBUG:root:i=647 residual=1.4990132513048593e-06
   DEBUG:root:i=648 residual=1.4827334098299616e-06
   DEBUG:root:i=649 residual=1.4663063439002144e-06
   DEBUG:root:i=650 residual=1.450685090276238e-06
   DEBUG:root:i=651 residual=1.4358116686707945e-06
   DEBUG:root:i=652 residual=1.4208330867404584e-06
   DEBUG:root:i=653 residual=1.4048738421479356e-06
   DEBUG:root:i=654 residual=1.391821456309117e-06
   DEBUG:root:i=655 residual=1.3782215546598309e-06
   DEBUG:root:i=656 residual=1.3699118426302448e-06
   DEBUG:root:i=657 residual=1.3487037904269528e-06
   DEBUG:root:i=658 residual=1.3336522215467994e-06
   DEBUG:root:i=659 residual=1.3180908808863023e-06
   DEBUG:root:i=660 residual=1.3048515938862693e-06
   DEBUG:root:i=661 residual=1.2985792636754923e-06
   DEBUG:root:i=662 residual=1.2782684279954992e-06
   DEBUG:root:i=663 residual=1.258791598957032e-06
   DEBUG:root:i=664 residual=1.2444884305296e-06
   DEBUG:root:i=665 residual=1.2352962812656187e-06
   DEBUG:root:i=666 residual=1.2199151342429104e-06
   DEBUG:root:i=667 residual=1.2070403272446129e-06
   DEBUG:root:i=668 residual=1.1933449286516407e-06
   DEBUG:root:i=669 residual=1.1830743460450321e-06
   DEBUG:root:i=670 residual=1.1674502502501127e-06
   DEBUG:root:i=671 residual=1.157635892923281e-06
   DEBUG:root:i=672 residual=1.1462703923825757e-06
   DEBUG:root:i=673 residual=1.129818997469556e-06
   DEBUG:root:i=674 residual=1.1198078482266283e-06
   DEBUG:root:i=675 residual=1.1054767128371168e-06
   DEBUG:root:i=676 residual=1.0934396641459898e-06
   DEBUG:root:i=677 residual=1.0816888789122459e-06
   DEBUG:root:i=678 residual=1.070140228875971e-06
   DEBUG:root:i=679 residual=1.0594490049697924e-06
   DEBUG:root:i=680 residual=1.0482540346856695e-06
   DEBUG:root:i=681 residual=1.0400917744846083e-06
   DEBUG:root:i=682 residual=1.0285316420777235e-06
   DEBUG:root:i=683 residual=1.0140994390894775e-06
   DEBUG:root:i=684 residual=1.0032105137725011e-06
   DEBUG:root:i=685 residual=9.917929446601192e-07
   INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
   INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
   INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
   INFO:root:rank=3 pagerank=3.1757e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
   INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
   INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
   INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
   INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
   INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
   INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
   ```

   Task 2, part 1:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
   INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
   INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
   INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
   INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
   INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
   INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
   INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
   INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
   INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
   INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
   ```

   Task 2, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
   INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
   INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
   INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
   INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
   INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
   INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
   INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
   INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
   INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
   INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-committee-holds-hearing-priorities-missile-defense
   ```

1. Ensure that all your changes to the `pagerank.py` and `README.md` files are committed to your repo and pushed to github.

1. Get at least 5 stars on your repo.
   (You may trade stars with other students in the class.)

   > **NOTE:**
   > 
   > Recruiters use github profiles to determine who to hire,
   > and pagerank is used to rank user profiles and projects.
   > Links in this graph correspond to who has starred/followed who's repo.
   > By getting more stars on your repo, you'll be increasing your github pagerank, which increases the likelihood that recruiters will hire you.
   > To see an example, [perform a search for `data mining`](https://github.com/search?q=data+mining).
   > Notice that the results are returned "approximately" ranked by the number of stars,
   > but because "some stars count more than others" the results are not exactly ranked by the number of stars.
   > (I asked you not to fork this repo because forks are ranked lower than non-forks.)
   >
   > In some sense, we are doing a "dual problem" to data mining by getting these stars.
   > Recruiters are using data mining to find out who the best people to recruit are,
   > and we are hacking their data mining algorithms by making those algorithms select you instead of someone else.
   >
   > If you're interested in exploring this idea further, here's a python tutorial for extracting GitHub's social graph: <https://www.oreilly.com/library/view/mining-the-social/9781449368180/ch07.html> ; if you're interested in learning more about how recruiters use github profiles, read this Hacker News post: <https://news.ycombinator.com/item?id=19413348>.

1. Submit the url of your repo to sakai.

   The assignment is worth 8 points.
   1. There are 6 parts to the output above.  (4 in Task1 and 2 in Task2.)
   1. Each part that you get incorrect will result in -2 points.  (But you cannot go negative.)
   1. Another way of phrasing this is that the first 2 parts you complete are not worth any points,
      but each part after that is worth 2 points.
