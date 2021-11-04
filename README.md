# TagsHasher

Inspired by the idea of `FeatureHasher`, `TagsHasher` converts arbitrary `.tags` into a fixed-length vector.

Note that, unlike `FeatureHashser`, you should only use Jaccard/Hamming distance when searching documents embedded via `TagsHasher`. This is because the closeness of the value on each feature is meaningless, as the value is the result of a hash function. Whereas in `FeatureHashser`'s example, the value represents the term frequency of a word. 

Hence, in `TagsHasher` only identity value in the embedded vector matters.

This demo requires `jina>=2.2.5.dev4`, if you encounter error try latest master.

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


```text
my filter is: {
  "Subject": "Comedy"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Chase, Chevy",
  "Actress": "",
  "Awards": "No",
  "Director": "",
  "Length": "",
  "Popularity": "82",
  "Subject": "Comedy",
  "Title": "Valkenvania",
  "Year": "1990"
}
{
  "*Image": "paulNewman.png",
  "Actor": "Newman, Paul",
  "Actress": "",
  "Awards": "No",
  "Director": "",
  "Length": "",
  "Popularity": "28",
  "Subject": "Comedy",
  "Title": "Secret War of Harry Frigg, The",
  "Year": "1968"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Murphy, Eddie",
  "Actress": "",
  "Awards": "No",
  "Director": "",
  "Length": "",
  "Popularity": "56",
  "Subject": "Comedy",
  "Title": "Best of Eddie Murphy, Saturday Night Live, The",
  "Year": "1989"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Mastroianni, Marcello",
  "Actress": "",
  "Awards": "No",
  "Director": "Fellini, Federico",
  "Length": "",
  "Popularity": "29",
  "Subject": "Comedy",
  "Title": "Ginger & Fred",
  "Year": "1993"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Piscopo, Joe",
  "Actress": "",
  "Awards": "No",
  "Director": "",
  "Length": "60",
  "Popularity": "14",
  "Subject": "Comedy",
  "Title": "Joe Piscopo New Jersey Special",
  "Year": "1987"
}


my filter is: {
  "Year": 1987.0
}
{
  "*Image": "NicholasCage.png",
  "Actor": "",
  "Actress": "Madonna",
  "Awards": "No",
  "Director": "",
  "Length": "50",
  "Popularity": "75",
  "Subject": "Music",
  "Title": "Madonna Live, The Virgin Tour",
  "Year": "1987"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Piscopo, Joe",
  "Actress": "",
  "Awards": "No",
  "Director": "",
  "Length": "60",
  "Popularity": "14",
  "Subject": "Comedy",
  "Title": "Joe Piscopo New Jersey Special",
  "Year": "1987"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Everett, Rupert",
  "Actress": "",
  "Awards": "No",
  "Director": "",
  "Length": "95",
  "Popularity": "25",
  "Subject": "Drama",
  "Title": "Hearts of Fire",
  "Year": "1987"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Lambert, Christopher",
  "Actress": "Sukowa, Barbara",
  "Awards": "No",
  "Director": "Cimino, Michael",
  "Length": "",
  "Popularity": "41",
  "Subject": "Drama",
  "Title": "Sicilian, The",
  "Year": "1987"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Hubley, Whip",
  "Actress": "",
  "Awards": "No",
  "Director": "Rosenthal, Rick",
  "Length": "98",
  "Popularity": "87",
  "Subject": "Action",
  "Title": "Russkies",
  "Year": "1987"
}


my filter is: {
  "Subject": "Comedy",
  "Year": 1987.0
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Piscopo, Joe",
  "Actress": "",
  "Awards": "No",
  "Director": "",
  "Length": "60",
  "Popularity": "14",
  "Subject": "Comedy",
  "Title": "Joe Piscopo New Jersey Special",
  "Year": "1987"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Murphy, Eddie",
  "Actress": "",
  "Awards": "No",
  "Director": "Murphy, Eddie",
  "Length": "90",
  "Popularity": "51",
  "Subject": "Comedy",
  "Title": "Eddie Murphy Raw",
  "Year": "1987"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "McCarthy, Andrew",
  "Actress": "Cattrall, Kim",
  "Awards": "No",
  "Director": "Gottlieb, Michael",
  "Length": "",
  "Popularity": "23",
  "Subject": "Comedy",
  "Title": "Mannequin",
  "Year": "1987"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Williams, Robin",
  "Actress": "",
  "Awards": "No",
  "Director": "Levinson, Barry",
  "Length": "120",
  "Popularity": "37",
  "Subject": "Comedy",
  "Title": "Good Morning, Vietnam",
  "Year": "1987"
}
{
  "*Image": "NicholasCage.png",
  "Actor": "Boys, The Fat",
  "Actress": "",
  "Awards": "No",
  "Director": "Schultz, Michael",
  "Length": "86",
  "Popularity": "69",
  "Subject": "Comedy",
  "Title": "Disorderlies",
  "Year": "1987"
}

Process finished with exit code 0
```
