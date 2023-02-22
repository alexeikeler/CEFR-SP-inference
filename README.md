# Updated run commands
Concat the two datasets:
```shell
mkdir CEFR-SP/all
cat CEFR-SP/SCoRE/CEFR-SP_SCoRE_train.txt CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_train.txt > CEFR-SP/all/all_train.txt
cat CEFR-SP/SCoRE/CEFR-SP_SCoRE_dev.txt CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_dev.txt > CEFR-SP/all/all_dev.txt
cat CEFR-SP/SCoRE/CEFR-SP_SCoRE_test.txt CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_test.txt > CEFR-SP/all/all_test.txt
```

```shell
python level_estimator.py --model bert-base-cased --lm_layer 11 --seed 935 --num_labels 6 --batch 128 --warmup 0 --with_loss_weight --num_prototypes 3 --type contrastive --init_lr 1.0e-5 --alpha 0.2 --data ../CEFR-SP/SCoRE/CEFR-SP_SCoRE --test ../CEFR-SP/SCoRE/CEFR-SP_SCoRE --out ../out/
```



# CEFR-Based Sentence Difficulty Annotation and Assessment

CEFR-SP provides 17k English sentences annotated with CEFR levels assigned by English-education professionals. 
For details of the corpus creation process and our CEFR-level assessment model, please refer to our [paper](https://arxiv.org/abs/2210.11766).

The CEFR-SP corpus is in `/CEFR-SP` directory and our codes for CEFR-level assessment model are in `/src` directory.  
Please refer to README of each directory for details.

## Citation
Please cite the following paper if you use the above resources for your research.
 ```
 Yuki Arase, Satoru Uchida, and Tomoyuki Kajiwara. 2022. CEFR-Based Sentence-Difficulty Annotation and Assessment. 
 in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2022) (Dec. 2022).
 
@inproceedings{arase:emnlp2022,
    title = "{CEFR}-Based Sentence-Difficulty Annotation and Assessment",
    author = "Arase, Yuki  and Uchida, Satoru, and Kajiwara, Tomoyuki",
    booktitle = "Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)",
    month = dec,
    year = "2022",
}
 ```

## Contact
Yuki Arase (arase [at] ist.osaka-u.ac.jp) 
-- please replace " [at] " with an "@" symbol.
