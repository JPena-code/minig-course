import pandas as pd
from pandas import CategoricalDtype

# Setting values of categorical attribute
# for performance at processing time
# Age numerical categories
AGE_RANGE = [
    '0--8',
    '9--17',
    '18--20',
    '21--23',
    '24--26',
    '27--29',
    '30--38',
    '39--47',
    '48+'
]
# Gender categories
GENDER_VALUES = [
    'Male',
    'Female',
    'Transgender/NonConforming',
]
# Age status categories
AGE_CATE = [
    'Adult',
    'Minor',
]
# Categorical objects attributes
CATEGORICAL_COLUMNS = [
    'ageBroad',
    'gender',
    'majorityStatus',
    'majorityStatusAtExploit',
    'majorityEntry',
    'citizenship',
    'CountryOfExploitation']

# Categorical pandas objects
AGE_CAT_RANGE = CategoricalDtype(categories=AGE_RANGE, ordered=True)
GENDER_CAT = CategoricalDtype(categories=GENDER_VALUES, ordered=False)
AGE_CAT = CategoricalDtype(categories=AGE_CATE, ordered=True)

# Majority Status Columns
COLUMNS_MAJORITY = [
    'majorityStatus',
    'majorityStatusAtExploit',
    'majorityEntry']
# Categorical countries
COUNTRY_COLUMNS = [
    'citizenship',
    'CountryOfExploitation']
# Concatenated columns names
CONCATENATED_COLUMNS = [
    'meansOfControlConcatenated',
    'typeOfExploitConcatenated',
    'typeOfLabourConcatenated',
    'typeOfSexConcatenated',
    'RecruiterRelationship']
# Mapping features with a shorter label
MAPPER_FEATURES = {
    'ageBroad': 'Age {}',
    'majorityStatus': 'Status {}',
    'majorityStatusAtExploit': 'At Exploit {}',
    'majorityEntry': 'Entry {}',
    'citizenship': 'Citizenship {}',
    'CountryOfExploitation': 'Exploitation {}'
}
