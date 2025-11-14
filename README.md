# segmentation-strategies
The code in the file Segmentation_strategies.ipynb allows for the evaluations of the 4 different segmentation strategies applied in the article. 

• Chunking : splits the text into non-overlapping
k-token blocks. It is fast and trivial, but loses entities
cut at boundaries and does not offer global context.
This approach also defines segments regardless of their
content, whether syntactically (sentences) or semantically
(change of thematic).

• Sliding window : as simple chunking, uses an iden-
tical block size completed with an α-token overlap. This
overlap is used to recover boundary entities and ensure
that their context is embedded in the segment.

• Thematic segmentation : data-driven topic
boundaries keep segments coherent and adaptive in length, but remains highly dependent to
the precision of boundary detection step whose errors can
hurt further model performances.

• Thematic segmentation with passage retrieval: ranks thematic segments with a dense retriever and
keep the top K. It offers the strongest coverage when
context is fragmented, at the cost of an index and added
latency but some entities can be losed.

The other files contain the dependency functions.

# Silver annotation
The Silver annotation file contains the code that uses GLiNER and NuNER on which pseudo-labeling has been applied to generate a new dataset of territorial food systems called consensus dataset or Silver dataset as shown on the fig below.

<img width="800" height="700" alt="meth (3)" src="https://github.com/user-attachments/assets/5001b91d-91da-45f5-acee-f7b1de5d08de" />



