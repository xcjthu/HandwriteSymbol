import logging
from .ImgFormatter import ImgFormatter
from .NewsRec.NewsRecFormatter import NewsRecFormatter
from .NewsRec.NewsRecTestFormatter import NewsRecTestFormatter
logger = logging.getLogger(__name__)

formatter_list = {
    "img": ImgFormatter,
    "NewsRec": NewsRecFormatter,
    "NewsRecTest": NewsRecTestFormatter,
}


def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logger.error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError
