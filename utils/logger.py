import logging

level = logging.INFO

# Create the logger
logger = logging.getLogger(__name__)
logger.setLevel(level)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(level)

# Create a log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
