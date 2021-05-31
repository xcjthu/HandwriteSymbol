import logging

from .Basic import BasicFormatter
from .RTEFormatter import RTEFormatter
from .RTEPromptFormatter import RTEPromptFormatter
from .RTEPromptRobertaFormatter import RTEPromptRobertaFormatter
from .SST2PromptFormatter import SST2PromptFormatter
from .SST2PromptRobertaFormatter import SST2PromptRobertaFormatter
from .WikiREFormatter import WikiREFormatter
from .WikiREPromptFormatter import WikiREPromptFormatter
logger = logging.getLogger(__name__)

formatter_list = {
    "Basic": BasicFormatter,
    "RTE": RTEFormatter,
    "RTEPrompt": RTEPromptFormatter,
    "RTEPromptRoberta": RTEPromptRobertaFormatter,
    "SST2Prompt": SST2PromptFormatter,
    "SST2_PromptRoberta": SST2PromptRobertaFormatter,
    "RE": WikiREFormatter,
    "REPrompt": WikiREPromptFormatter,
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
