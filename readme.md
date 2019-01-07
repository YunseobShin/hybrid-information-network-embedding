Preparing dataset:

1. go to https://dumps.wikimedia.org/backup-index.html and choose proper language and date.

2. download xml.bz2 file.

3. extract files

4. you can combine multiple article chunks by cat command e.g. '''cat ./articles1 article2 > '''



To use wiki extractor:

```python eikiextractor/WikiExtractor wiki.xml```

Then, combine the extracted parts into one.

```cat text/*/* > wiki.txt```

example:
```python example.py --xml wiki.xml --txt wiki.txt --edge edge.txt --sample_size 5000 --alpha 0.5```
