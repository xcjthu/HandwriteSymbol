from .ImageFromJson import ImageDataset
from .NewsRec.NewsRecDataset import NewsRecDataset, NewsRecTestDataset
dataset_list = {
    "img": ImageDataset,
    "NewsRec": NewsRecDataset,
    "NewsRecTest": NewsRecTestDataset,
}
