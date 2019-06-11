### Dependencies
* Install Python 3
* Install Pip3: sudo apt install python3-pip
* Install requirements: pip3 install -r requirements.txt
* [Faiss](https://github.com/facebookresearch/faiss) (recommended) for fast nearest neighbor search (CPU or GPU).

Available on CPU or GPU, in Python 2 or 3. Faiss is *optional* for GPU users - though Faiss-GPU will greatly speed up the nearest neighbor search - and *highly recommended* for CPU users. Faiss can be installed using "conda install faiss-cpu -c pytorch" or "conda install faiss-gpu -c pytorch".

### Run the code: adversarial training and refinement (CPU|GPU)
A sample command to learn a mapping using adversarial training and iterative Procrustes refinement:
```bash
python3 unsupervised.py --src_lang java --tgt_lang cs --src_emb data/java_vectors.txt --tgt_emb data/cs_vectors.txt --n_refinement 2 --emb_dim 50 --max_vocab 300000 --epoch_size 100000 --n_epochs 1 --identical_dict_path "dict/candidates_dict.txt" --dico_eval "eval/java-cs.txt"
```
### Evaluate cross-lingual embeddings (CPU|GPU)

```bash
python3 evaluate.py --src_lang java --tgt_lang cs --src_emb dumped/debug/some_id/vectors-java.txt --tgt_emb dumped/debug/some_id/vectors-cs.txt --dico_eval "eval/java-cs.txt" --max_vocab 200000
```