from .ImageFromJson import ImageDataset
from .NewsRec.NewsRecDataset import NewsRecDataset
dataset_list = {
    "img": ImageDataset,
    "NewsRec": NewsRecDataset
}
