# Notice

This folder contains survey papers we used from arixv, and cited papers of them are extracted with title and abstract for retrieval and generation.

**survey_info.txt**: the first col is the arixv id and second col is the title of the survey.

**trees** folder: pyg graph where the first node is a survey and the other nodes are references to the survey.

**graphs** folder: pyg The graph is actually a 2-hop ego-graph for each survey, and the central node (the survey) has been removed.

Some surveys may be documented in survey_info.txt, but have no corresponding figures because the bib or bbl files cannot be extracted.
