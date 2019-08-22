# knowledge_augmented_bert

PS: text here means search query from the input of users.

Pretrain bert with text-classification task and traditional masked word prediction task. In this way, text embedding with knowledge level augmentation can be derived from the specificly designed BERT.

![image](https://github.com/CristinaMa0917/knowledge_augmentation_bert/blob/unify_task/images/img1.png)

Exactly two different pretrain tasks are designed which are parallel task and unified task. The only differnce is the usage of the output bert embeddings.
The finetune preprocess is shown as follows.

![image](https://github.com/CristinaMa0917/knowledge_augmentation_bert/blob/unify_task/images/img2.png)

Through multiple comparisons, Unified Text encoding from BERT (UTEB) is better than Parallel Text encoding from BERT(PTEB). Besides UTEB shows faster convergence and higher accuracy than TextCNN.

![image](https://github.com/CristinaMa0917/knowledge_augmentation_bert/blob/unify_task/images/img3.png)
