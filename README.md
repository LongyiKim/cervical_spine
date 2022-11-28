# cervical_spine
RSNA 2022 Cervical Spine Fracture Detection (Kaggle)

The whole developing process for this competetion. Finally I got 22nd in the private score, but TOO MUCH mistakes from the very begining.
I learned a lot from this project, and I am looking forward a better score in other competitions. 

My final score method is here, 
  1. Segment each slices using Efficient Unet. The model outputs two value, one is the logits of background, the other is a 0-1 score to predict C1-C7. The label C value is setted as C1 -> 0.125, C2 -> 0.25, ..., C7 -> 0.875
  
  loss = BCE(Value 1, Background) + lambda * MSE(Value 2, C value), and the success point here is I setted positive weight as 50 in BCE loss function, and lambda setted 200.  
  
  2. Find fracture using EffDet Model. Using predicted bounding box, and sum up segmented result to find out which C part is inside the bounding box. 
  3. Just use maximum value as overall fracture score. 
  
  
I found that even though I had came up with some GOOD ideas, but I couldn't make it work.
For example, sementation using 3D input, RNN for overall score. 

This competetion told me that I should keep doing with CORRECT idea, not a possible idea. 
