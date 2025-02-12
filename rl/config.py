import os
import datetime
import faker

f = faker.Faker()

def get_random_word_string():
    return ''.join([f.word().capitalize() for _ in range(3)])

LOG_DIR_ROOT = 'logs/'
RUN_NAME = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{get_random_word_string()}"
MODELS_DIR = 'models/'