# LowPrecision

This project is low-precision related. Mainly focused on the decoder part of NMT system. 

Attempted to replace the float32 units commonly used in the neural network with lower precision units, such as float16, int, int8, etc. in order to improve the speed and save memory.

---

## Catalog

Catalog | Content
---|---
/BLUE |                     Python script for testing BLUE values
/LinerSoftmax |             Based on Neu. Tensor project, the source code of low-precision classifier experimented on FNNLM language model.
/LowPrecision/src |	        Write the conversion and calculation between semi-precision (float16) and other precision units in CPU environment
/NN/src |	                  Source code of FNNLM
/Note |	                    Note, problems with gpu coding
/NumberVerification/src |	  Source code of testing MINST
/RNN&Attention |	          Based on a seq2seq translation model of RNN & Attention, tested with low-precision decoder
/fairseq_work |	            Based on Facebook's fairseq translation model, tested with low-precision decoder 
/paper |                    Related paper
presentation.pdf |          Presentation

---

In this project, the main contents of the internship in the Natural Language Processing Laboratory (calf translation) of Northeast University from January 2019 to March 2019 have been sorted out. 

Thank all the teachers, seniors, classmates of the laboratory for their great help. And especially thank T.Xiao Tong and Y.Lin for their guidance and help in my experiment.

try something



