# Kingfisher video observation analysis
This repository documents efforts to analyze video observations of Kingfisher nesting sites.
Goal of the analysis is to effectively identify Kingfisher individuals approaching the nesting site.

# Appraoches taken
## Frame-wise detection of Kingfisher presence
The first approach tried is to detect the presence of Kingfishers in individual video frames, with no contextual information.
This is the simplest appraoch from a technical standpoint and if successful, suffices to fully achieve the goal.

The task is framed as an binary image classfiication task, with the presence of a Kingfisher being the positive class.
Contiguous sequences of video frames at a single nesting site were labelled manually by a domain expert, to enable a supervised learning approach.
We use Tensorflow as the deep learning framework of choice.
