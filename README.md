# TagsHasher

Inspired by the idea of `FeatureHasher`, `TagsHasher` converts arbitrary `.tags` into a fixed-length vector.

Note that, unlike `FeatureHashser`, you should only use Jaccard/Hamming distance when searching documents embedded via `TagsHasher`. This is because the closeness of the value on each feature is meaningless, as the value is the result of a hash function. Whereas in `FeatureHashser`'s example, the value represents the term frequency of a word. 

Hence, in `TagsHasher` only identity value in the embedded vector matters.

## Example

I will keep everything out of the Flow to make it clear:

```python
import io

from executor import TagsHasher
from jina import Document, DocumentArray
from jina.types.document.generators import from_csv

# Load some online CSV file dataset
src = Document(
    uri='https://perso.telecom-paristech.fr/eagan/class/igr204/data/film.csv'
).convert_uri_to_text('iso8859')
da = DocumentArray(from_csv(io.StringIO(src.text), dialect='auto'))

# use TagsHasher to encode data
th = TagsHasher()
th.encode(da)

# build some filters
filters = [
    {"Subject": "Comedy"},
    {"Year": 1987},
    {"Subject": "Comedy", "Year": 1987}
]

# build Documents from filters
qa = DocumentArray([Document(tags=f) for f in filters])

# and then use TagsHasher to encode them
th.encode(qa)

# do match, show top-5, notice the usage of Jaccard here. It requires scipy as jaccard is not natively supported by Jina
qa.match(da, limit=5, exclude_self=True, metric='jaccard', use_scipy=True)

# print
for d in qa:
    print('my filter is:', d.tags.json())
    for m in d.matches:
        print(m.tags.json())
    input()

```

