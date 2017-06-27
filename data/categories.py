
from __future__ import print_function

import os
import json
import pickle
import hashlib
import collections

import editdistance

dir = os.path.dirname(os.path.abspath(__file__))



class CategoryDeterminer:
    category_mapper_filepath = os.path.join(dir, 'umls/category2aliases.pkl')
    category_dict_filepath = os.path.join(dir, 'umls/category_dict.json')
    category_folder = os.path.join(dir, 'umls/categories')

    def __init__(self, cui2info_filepath='umls/cui2info.pkl', verbose=False):
        self.verbose = verbose

        if not os.path.isfile(self.category_mapper_filepath):
            self._print('Gathering categories..')

            with open(os.path.join(dir, cui2info_filepath), 'rb') as f:
                data = pickle.load(f)

            categories = list(data[0].keys())
            category2aliases = {cat: [] for cat in categories}

            for _, value in data[1].items():
                for category in map(self.clean_alias, value['sem_typ']):
                    for alias in map(self.clean_alias, value['aliases']):
                        category2aliases[category].append(alias)

            # Save categories to text files
            if not os.path.isdir(self.category_folder):
                os.mkdir(self.category_folder)

            for category, aliases in category2aliases.items():
                category_filename = os.path.join(self.category_folder, '%s.txt' % (category))
                with open(category_filename, 'w') as f:
                    for alias in sorted(aliases):
                        f.write('%s\n' % (alias))

            with open(self.category_mapper_filepath, 'wb') as f:
                pickle.dump(category2aliases, f)

        # Load category mapper
        with open(self.category_mapper_filepath, 'rb') as f:
            self.category2aliases = pickle.load(f)

        # Load category dict
        try:
            with open(self.category_dict_filepath, 'r') as f:
                self.category_dict = json.load(f)
        except FileNotFoundError:
            self.category_dict = {}

    def alias_iterator(self):
        for category, aliases in self.category2aliases.items():
            for alias in aliases:
                yield category, alias

    def clean_alias(self, string):
        return string.lower().strip()

    def define_categories(self, categories):
        new_categories = set()
        for category in categories:
            new_categories.add(self.lookup_category(category))

        return list(sorted(new_categories))

    def lookup_category(self, category):
        category = self.clean_alias(category)
        if category in self.category_dict:
            return self.category_dict[category]

        # Find match and add to category dict
        category_match = self.find_match(query=category)
        _category, _alias, _dist = category_match
        self.category_dict[category] = _category

        # Save category dict
        with open(self.category_dict_filepath, 'w') as f:
            json.dump(self.category_dict, f, indent=4, sort_keys=True)

        return category_match

    def find_match(self, query):
        query = self.clean_alias(query)

        # First check if the query is a substring of any of the aliases
        # (only if the query is at least 5 characters long)
        query_length = len(query)
        lower_length, upper_length = query_length - 5, query_length + 5
        if query_length >= 5:
            for category, alias in self.alias_iterator():
                alias_length = len(alias)
                if alias_length < lower_length: continue
                if alias_length > upper_length: continue
                if query in alias or alias in query:
                    return (category, alias, editdistance.eval(query, alias))

        # If any alias have a distance below the following it is considered
        # a match as well
        max_dist = min(query_length // 5, 5)

        closest_match = ('<no_category>', '<no_match>', float('inf'))
        for category, alias in self.alias_iterator():
            alias_length = len(alias)
            if alias_length < lower_length: continue
            if alias_length > upper_length: continue

            dist = editdistance.eval(query, alias)
            if dist < closest_match[2]:
                closest_match = (category, alias, dist)

                if dist <= max_dist:
                    return closest_match
        return closest_match

    def _print(self, message):
        if self.verbose:
            print(message)




# if __name__ == '__main__':

#     # with open(os.path.join(dir, 'cui2info.pkl'), 'rb') as f:
#     #     data = pickle.load(f)



#     # categories = list(data[0].keys())
#     # category2aliases = {cat: [] for cat in categories}



#     # def clean_alias(string):
#     #     return string.lower().strip()


#     # for key, value in data[1].items():
#     #     for category in map(clean_alias, value['sem_typ']):
#     #         for alias in map(clean_alias, value['aliases']):
#     #             category2aliases[category].append(alias)


#     # category_folder = os.path.join(dir, 'categories')
#     # if not os.path.isdir(category_folder):
#     #     os.mkdir(category_folder)


#     # for category, aliases in category2aliases.items():
#     #     category_filename = os.path.join(category_folder, '%s.txt' % (category))
#     #     with open(category_filename, 'w') as f:
#     #         for alias in sorted(aliases):
#     #             f.write('%s\n' % (alias))


#     # def alias_iterator():
#     #     for category, aliases in category2aliases.items():
#     #         for alias in aliases:
#     #             yield category, alias


#     # def find_match(query, dist_func=levenshtein):
#     #     query = clean_alias(query)

#     #     # First check if the query is a substring of any of the aliases
#     #     # (only if the query is at least 5 characters long)
#     #     query_length = len(query)
#     #     lower_length, upper_length = query_length - 5, query_length + 5
#     #     if query_length >= 5:
#     #         for category, alias in alias_iterator():
#     #             alias_length = len(alias)
#     #             if alias_length < lower_length: continue
#     #             if alias_length > upper_length: continue
#     #             if query in alias:
#     #                 return (category, alias, dist_func(query, alias))

#     #     # If any alias have a distance below the following it is considered
#     #     # a match as well
#     #     max_dist = min(query_length // 5, 5)

#     #     closest_match = ('<no_category>', '<no_match>', float('inf'))
#     #     for category, alias in alias_iterator():
#     #         alias_length = len(alias)
#     #         if alias_length < lower_length: continue
#     #         if alias_length > upper_length: continue

#     #         dist = dist_func(query, alias)
#     #         if dist < closest_match[2]:
#     #             closest_match = (category, alias, dist)
#     #             print(closest_match)

#     #             if dist <= max_dist:
#     #                 return closest_match
#     #     return closest_match



#     category_determiner = CategoryDeterminer(verbose=True)

#     closest_match = category_determiner.find_match(query='absent kidney')
#     print('closest_match')
#     print(closest_match)
#     closest_match = category_determiner.find_match(query='abses kidney')
#     print('closest_match')
#     print(closest_match)






# Define category mappings:
# * First value of tuple is the true origin.
# * Second value of tuple is a list of category names which maps to the
#   origin category.
_category_mappings = [
    ('acne',                        ['adult acne', 'blackhead', 'whitehead']),
    ('abdominal',                   ['abdominal hernia', 'abdominal muscles', 'abdominal pain']),
    ('acid',                        ['uric acid']),
    ('alcohol',                     ['alcohol abuse', 'alcohol withdrawal', 'alcoholism']),
    ('allergy',                     ['allergic asthma', 'allergic reaction', 'food allergy']),
    ('anatomy',                     ['arm', 'leg', 'finger', 'foot', 'armpit', 'back pain', 'ear infection', 'ears', 'elbow', 'hand', 'head', 'hair', 'hair loss', 'lip', 'mouth', 'nail', 'neck', 'nipple', 'nose', 'runny nose', 'shoulder', 'stomach', 'stuffy nose', 'toe', 'tongue', 'wrist', 'wrist pain', 'chest', 'forehead', 'jaw', 'knee', 'spine', 'ankle', 'belly button', 'pelvis', 'thigh', 'throat', 'tooth', 'wisdom tooth', 'broken bone', 'cervix', 'rectum', 'scalp', 'scrotum', 'thyroid', 'torso', 'anus', 'big toe', 'calf', 'eardrum', 'ear drum', 'eyebrow', 'forearm', 'heel', 'knee pain', 'shin', 'skull', 'thumb', 'waist', 'buttocks', 'nostril']),
    ('bacteria',                    ['bacterial pneumonia', 'bacterial vaginosis', 'bacterium']),
    ('baby',                        ['infant', 'toddler', 'diaper rash', 'newborn', 'teething']),
    ('bone disease',                ['osteoarthritis', 'osteoporosis']),
    ('blood',                       ['blood clot', 'blood pressure', 'blood sugar', 'blood test', 'sepsis', 'blood thinner', 'high blood pressure', 'low blood pressure', 'hemoglobin', 'circulation', 'anemia']),
    ('body fat',                    ['triglyceride', 'cholesterol', 'calorie']),
    ('cancer',                      ['uterine cancer', 'prostate cancer', 'cervical cancer', 'chemotherapy', 'colon cancer', 'paget\'s disease', 'lung cancer', 'skin cancer', 'stomach cancer', 'breast cancer']),
    ('chemical',                    ['sodium', 'lithium', 'oxygen']),
    ('child disease',               ['chickenpox', 'measles']),
    ('cold',                        ['cold sore', 'coldness']),
    ('diabetes',                    ['type 1 diabetes', 'type 2 diabetes']),
    ('drugs',                       ['antibiotic', 'antibiotic use', 'antidepressant', 'aspirin', 'drug', 'drug overdose', 'drug test', 'ibuprofen', 'inhaler', 'insulin', 'marijuana', 'penicillin', 'steroid', 'depo-provera', 'epidural', 'injection', 'lotion', 'beta blocker', 'diet pills', 'morphine', 'narcotic', 'nasal spray', 'sunscreen', 'oil', 'cocaine', 'cephalexin', 'caffeine', 'addiction', 'painkiller', 'amoxicillin']),
    ('exercise',                    ['exercise stress test', 'walking', 'movement', 'running', 'workout', 'athlete\'s foot', 'bruise', 'fitness', 'high cholesterol', 'injury', 'metabolism', 'sprain', 'yoga', 'obesity']),
    ('financial',                   ['aca', 'affordable care act', 'insurance', 'medicare', 'obamacare', 'over the counter', 'health insurance', 'health insurance exchange', 'health insurance marketplace', 'medicaid', 'healthcare it']),
    ('food',                        ['turkey', 'tuna', 'diet', 'fish', 'milk', 'orange', 'salt', 'supplement', 'tea', 'wine', 'coffee', 'healthy eating', 'hunger', 'juice', 'meal', 'lactose intolerance', 'cereal', 'cooking', 'craving', 'fiber', 'fruit', 'garlic', 'herb', 'honey', 'lemon', 'salad', 'oatmeal', 'rice', 'protein', 'shellfish', 'soda', 'syrup', 'vegetable', 'vinegar', 'yogurt', 'peanut', 'pepper', 'olive', 'meat', 'snack', 'nutrition', 'taste']),
    ('genitals',                    ['genital herpes', 'genitalia', 'gonorrhea', ]),
    ('headache disorder',           ['migraine', 'headache', 'headaches', 'tinnitus', 'concussion', 'confusion', 'nausea', 'vertigo']),
    ('heart',                       ['heart attack', 'heart disease', 'heartburn', 'stroke', 'congestive heart failure', 'transient ischemic attack', 'transient ischemia attack', 'heart murmur', 'seizure', 'tachycardia', 'pea']),
    ('height',                      ['short stature']),
    ('hormone',                     ['testosterone', 'estrogen', 'cortisone']),
    ('infection',                   ['hepatitis'] + ['hepatitis %s' % (char) for char in 'abc'] + ['malaria', 'strep throat', 'tetanus', 'toxoplasmosis', 'yeast', 'yeast infection', 'trichomoniasis', 'thrush', 'staph infection', 'pancreatitis', 'acute pancreatitis']),
    ('inflammation',                ['tendonitis']),
    ('lung disease',                ['chronic obstructive pulmonary disease', 'emphysema', 'pneumonia', 'tuberculosis', 'asthma', 'asthma attack', 'bronchitis', 'acute bronchitis', 'cough', 'whooping cough', 'cystic fibrosis']),
    ('mental disorder',             ['bipolar disorder', 'dementia', 'fibromyalgia', 'obsessive-compulsive disorder', 'psychosis', 'schizophrenia', 'alzheimer\'s disease', 'anxiety', 'attention deficit hyperactivity disorder', 'autism', 'epilepsy', 'trauma', 'mood swing', 'suicide', 'anesthesia', 'anger', 'sleepiness', 'paranoid behavior']),
    ('menstrual',                   ['menopause', 'spotting', 'cramps', 'post-menopause']),
    ('nerve disease',               ['parkinson\'s disease']),
    ('organ',                       ['kidney', 'liver', 'lung', 'sinus', 'sinus infection', 'artery', 'gallbladder', 'gallstone', 'muscle', 'nerve', 'vein', 'intestine', 'kidney disease', 'kidney infection', 'kidney stone', 'tonsil', 'hernia', 'hiatal hernia']),
    ('pain',                        ['sore throat', 'sting', 'prostatitis', 'sore', 'chronic pain', 'stomach pain', 'arthritis', 'tenderness']),
    ('penis',                       ['foreskin', 'testicle', 'undescended testicle', 'erectile dysfunction', 'circumcision', 'prostate']),
    ('pets',                        ['cat', 'cats', 'dog', 'dogs']),
    ('plants',                      ['poison ivy', 'plant', 'aloe', 'aloe vera']),
    ('pregnancy',                   ['abortion', 'birth', 'birth control', 'birth control pill', 'birth defect', 'fertility', 'fetus', 'miscarriage', 'morning after pill', 'period', 'pregnancy test', 'prenatal vitamin', 'tampon', 'infertility', 'ovulation', 'morning sickness', 'pelvic area', 'breastfeed', 'ectopic pregnancy', 'first trimester', 'labor', 'missed period', 'pregnant']),
    ('sex',                         ['orgasm', 'sexual intercourse', 'sperm', 'condom', 'ejaculation', 'erection', 'masturbation', 'libido', 'premature ejaculation']),
    ('skin disease',                ['hives', 'urticaria', 'herpes', 'herpes simplex', 'skin', 'dry skin', 'ringworm', 'atopic dermatitis', 'blister', 'burn', 'mold', 'mole', 'rosacea', 'scar', 'psoriasis', 'rash', 'sunburn', 'tattoo', 'wart', 'wrinkle', 'pore', 'stitches']),
    ('smoking',                     ['quit smoking', 'nicotine', 'tobacco']),
    ('std',                         ['sexually transmitted disease', 'chlamydia', 'hiv']),
    ('stress',                      ['depression', 'postpartum depression']),
    ('treatments',                  ['biopsy', 'cat scan', 'radiation surgery', 'screening', 'surgery', 'ultrasound', 'vasectomy', 'x-ray', 'colonoscopy', 'endoscopy', 'hysterectomy', 'liposuction', 'mammogram', 'massage', 'stent', 'gastric bypass surgery', 'physical therapy', 'electrogastrogram', 'spinal fusion']),
    ('tumor',                       ['adenoma', 'fibroid', 'cyst', 'brain tumor', 'benign', 'benign prostatic hyperplasia']),
    ('urination',                   ['urinalysis', 'urinary incontinence', 'urinary tract', 'urinary tract infection', 'urethra', 'bladder', 'overactive bladder']),
    ('uterine',                     ['uterus', 'uterine prolapse', 'ovary']),
    ('vaginal problem',             ['vaginal yeast infection', 'vaginal discharge', 'vagina', 'clitoris', 'toxic shock syndrome']),
    ('vaccination',                 ['vaccines']),
    ('vision',                      ['blindness', 'blinking', 'eyes', 'eyelid', 'cataract', 'pink eye', 'sty']),
    ('vitamin',                     ['vitamin %s' % (char) for char in 'abcde'] + ['vitamin d deficiency', 'b12', 'calcium', 'iron', 'multivitamin', 'zinc']),
    ('virus',                       ['gastroenteritis', 'human papillomavirus', 'shingles', 'swine flu', 'mononucleosis']),
    ('vomit',                       ['vomiting blood']),
    ('weight',                      ['weight gain', 'weight loss'])
]
_category_mappings_dict = {}
for target, sources in _category_mappings:
    for source in sources:
        _category_mappings_dict[source] = target

# Make sure no word is both a source and target
_sources = set(source for source in _category_mappings_dict.keys())
_targets = set(target for target in _category_mappings_dict.values())
_intersection = _targets & _sources
assert _intersection == set(), '%s cannot be source and target' % (_intersection)

def lookup_category(category):
    if category in _category_mappings_dict:
        return _category_mappings_dict[category]
    return category

def valid_category(category):
    return category in _sources or category in _targets


# Descriptions of some categories:
# 'aca': 'Affordable Care Act (obamacare)'
#

def define_categories(data, N=100):
    """
        Define a "categories" key for each element (Q/A) in `data`, with its
        value being a list of categories the element is a part of.

        The used categories are found using the following approach:

        1) Map all defined sub-categories to main-categories
        2) Count category occurencies
        3) Extract `N` most common categories
        4) Filter away rare categories

        Returns the used categories.
    """

    # 1) Map all sub-categories to the main-catagories
    new_categories = set()
    for x in data:
        x['categories'] = list(map(lookup_category, x.get('tags', [])))

    # 2) Count category occurencies
    counter = collections.Counter()
    for x in data:
        for categories in x['categories']:
            counter[categories] += 1

    # 3) Extract N most common categories
    common_category_counts = counter.most_common(n=N)
    common_categories = set([category for category, count in common_category_counts])

    # 4) Filter away rare categories
    for x in data:
        x['categories'] = [category for category in x['categories'] if category in common_categories]
        x['categories'] = list(set(x['categories']))

    return list(sorted(common_categories))




if __name__ == '__main__':

    with open(os.path.join(dir, 'webmd/test-with-tags.json'), 'r') as f:
        data = json.load(f)

    tags = set()
    for x in data:
        for tag in x.get('tags', []):
            tags.add(tag)

    category_determiner = CategoryDeterminer(verbose=True)

    # Look up category mappings and return the used categories
    categories = category_determiner.define_categories(categories=tags)
    print('categories')
    print(categories)
    print(len(categories))


    import sys; sys.exit()

    import numpy as np
    import matplotlib.pyplot as plt

    N_values = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    index = np.arange(0, len(N_values))

    coverages = []
    for N in N_values:
        with open(os.path.join(dir, 'webmd/test-with-tags.json'), 'r') as f:
            data = json.load(f)

        categories = define_categories(data, N=N)

        count = 0
        for x in data:
            if len(x['categories']) > 0:
                count += 1
        coverage = count / len(data)
        coverages.append(coverage)
        print('Category coverage (N=%d): %.4f' % (N, coverage))

    plt.bar(index, coverages, width=1.0)
    plt.xlabel('N')
    plt.ylabel('Count')
    plt.xlim([index.min(), index.max() + 0.5])
    plt.xticks(index + 0.5, N_values)

    plt.show()
