
import os
import re
import csv
import json
import time
import math
import datetime
import collections
import requests
from urllib import parse
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

import numpy as np

from .data_loader import DataLoader
from data.utils import Alphabet, process_text, pad_sequence
from utils.strings import extract_sentences, truncate_by_word
from utils.strings import word_end_indices

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class WikiLoader(DataLoader):
    EXTRA_DATA_FOLDER = os.path.join(FILEPATH, 'extra')
    def __init__(self, alphabet,
                 main_category_url='/wiki/Portal:Contents/Health_and_fitness',
                 **kwargs):
        """ TODO: Class description.
        """
        # Define data folder
        self.FOLDER = os.path.join(FILEPATH, 'wiki')
        self.TEXT_FILE = os.path.join(self.FOLDER, 'processed-text.csv')

        # Save wikipedia categories to file
        self.wiki_category_crawler = WikiCategoryCrawler(
            folder=self.FOLDER,
            main_category_url=main_category_url
        )

        # Parse wikipedia xml file to csv
        self.wiki_xml_parser = WikiXMLParser(folder=self.FOLDER)

        # Define alphabet
        self.alphabet = alphabet

        # Run parent constructor
        super().__init__(**kwargs)

        # Overwrite max_words
        self.max_words = self.max_dec_seq_length // 4

        # dtype used for specifying data-type in the memmap
        self.dtype = 'int8'

        # Define data row keys and indices
        last_idx = 0
        self.data_indices = {}
        for key, length in [
            ('enc_input',            self.max_enc_seq_length),     # Enc input
            ('dec_input',            self.max_dec_seq_length + 1), # Dec output
            ('dec_target',           self.max_dec_seq_length + 1), # Dec target
            ('enc_input_length',     1),                           # Input length
            ('dec_input_length',     1),                           # Output length
            ('is_lm_mode',           1),                           # Binary mode value
            ('class_count',          1),                           # Number of associated classes
            ('classes',              self.max_classes),            # Associated classes
            ('input_word_features',  self.max_features),           # Word features
            ('target_word_features', self.max_features),           # Word features
            ('ws_indices',           self.max_words),              # Word indices
            ('word_count',           1),                           # Number of words
        ]:
            self.data_indices[key] = (last_idx, last_idx + length)
            last_idx += length

        # Number of total elements per observation
        self.row_elements = last_idx

        # Create or load data
        # Defines `self.train_data`, `self.val_data`, and `self.test_data`.
        self.setup_data()


    def setup_data(self):

        def valid_content(content):
            return len(content) > 2 and len(content) < 1000

        # # Plot sequence lengths
        # import matplotlib.pyplot as plt
        # seq_lengths = []
        # for i, content in enumerate(self.content_iterator()):
        #     if valid_content(content):
        #         seq_lengths.append(len(content))

        # fig = plt.figure(1)
        # plt.hist(seq_lengths)
        # fig.savefig('wiki-seq-lengths.pdf', bbox_inches='tight')

        # seq_lengths = np.asarray(seq_lengths)
        # seq_lengths = seq_lengths[abs(seq_lengths - np.mean(seq_lengths)) < 2 * np.std(seq_lengths)]

        # fig = plt.figure(2)
        # plt.hist(seq_lengths)
        # fig.savefig('wiki-seq-lengths-2.pdf', bbox_inches='tight')

        # Construct mappings if not already exists
        if not self.dependencies_constructed() or self.force_reconstruct:
            self._print('Constructing dependencies..')

            # Count data
            data_count = 0
            char_counter = collections.Counter()
            for content in self.content_iterator():
                if valid_content(content):
                    data_count += 1

            indices = np.random.permutation(data_count)
            test_count  = math.floor(self.test_fraction * data_count)
            val_count   = math.floor(self.val_fraction  * data_count)
            train_count = data_count - test_count - val_count
            indices_test  = set(indices[0:test_count])
            indices_val   = set(indices[test_count:test_count + val_count])
            indices_train = set(indices[test_count + val_count:])

            train_data_memmap = np.memmap(
                filename=self.train_memmap_filename,
                dtype=self.dtype,
                mode='w+',
                shape=(train_count, self.row_elements)
            )
            val_data_memmap = np.memmap(
                filename=self.val_memmap_filename,
                dtype=self.dtype,
                mode='w+',
                shape=(val_count, self.row_elements)
            )
            test_data_memmap = np.memmap(
                filename=self.test_memmap_filename,
                dtype=self.dtype,
                mode='w+',
                shape=(test_count, self.row_elements)
            )

            # Make sure to write to random row
            write_idx_train = np.random.permutation(train_count)
            write_idx_val   = np.random.permutation(val_count)
            write_idx_test  = np.random.permutation(test_count)

            data_idx = 0
            train_count = val_count = test_count = 0
            for content in self.content_iterator():
                if not valid_content(content):
                    continue

                # Truncate sentences (truncate by word)
                output = truncate_by_word(content, max_length=self.max_dec_seq_length)

                # Tokenize question and answer
                input_int  = [self.alphabet.PAD_ID] * self.max_enc_seq_length
                output_int = [self.alphabet.GO_ID] + self.alphabet.encode_seq(output)

                input_length = 0
                output_int, output_length = pad_sequence(output_int, max_length=self.max_dec_seq_length + 1, alphabet=self.alphabet)
                target_int                = output_int[1:] + [0]

                # Create word separation indices
                ws_indices = word_end_indices(string=content)
                ws_indices = ws_indices[:self.max_words]
                word_count = len(ws_indices)
                ws_indices += [self.alphabet.PAD_ID] * (self.max_words - word_count)

                if word_count == 0:
                    continue

                classes_vec = [-1] * self.max_classes
                classes_count = 0

                # Construct data row
                row  = input_int + output_int + target_int
                row += [input_length, output_length]
                row += [1] # For binary value "mode"
                row += [classes_count] # Number of classes
                row += classes_vec     # Question classes
                row += [0] * (2 * self.max_features) # Not relevant
                row += ws_indices
                row += [word_count]
                row  = np.asarray(row, dtype=self.dtype)

                # Append
                if data_idx in indices_val:
                    val_data_memmap[write_idx_val[val_count],:] = row
                    val_count += 1
                elif data_idx in indices_test:
                    test_data_memmap[write_idx_test[test_count],:] = row
                    test_count += 1
                else:
                    train_data_memmap[write_idx_train[train_count],:] = row
                    train_count += 1

                data_idx += 1
                if data_idx % 10000 == 0:
                    self._print('Processed %d observations' % (data_idx))

            # Close files
            train_data_memmap.flush()
            val_data_memmap.flush()
            test_data_memmap.flush()
            del train_data_memmap, val_data_memmap, test_data_memmap

            # Save meta data
            with open(self.meta_data_filename, 'w') as f:
                json.dump({
                    'dtype': self.dtype,
                    'row_elements': self.row_elements,
                    'counts': {
                        'total': data_count,
                        'train': train_count,
                        'val':   val_count,
                        'test':  test_count
                    },
                    'shapes': {
                        'train': (train_count, self.row_elements),
                        'val':   (val_count,   self.row_elements),
                        'test':  (test_count,  self.row_elements)
                    }
                }, f, indent=4)

        # Load meta data
        with open(self.meta_data_filename, 'r') as f:
            self.meta = json.load(f)

        if self.validation == 'train':
            self.data_memmap = np.memmap(
                filename=self.train_memmap_filename,
                dtype=self.meta['dtype'],
                mode='c',
                shape=tuple(self.meta['shapes']['train'])
            )
        elif self.validation == 'val':
            self.data_memmap = np.memmap(
                filename=self.val_memmap_filename,
                dtype=self.meta['dtype'],
                mode='c',
                shape=tuple(self.meta['shapes']['val'])
            )
        elif self.validation == 'test':
            self.data_memmap = np.memmap(
                filename=self.test_memmap_filename,
                dtype=self.meta['dtype'],
                mode='c',
                shape=tuple(self.meta['shapes']['test'])
            )
        else:
            raise KeyError('Invalid validation argument provided')


    def __iter__(self):
        for x in self.data_memmap:
            yield x


    def content_iterator(self):
        self._print('Loading wikipedia articles')
        with open(self.wiki_xml_parser.ARTICLES_CSV_FILENAME, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for title, content in reader:
                for sentence in extract_sentences(content):
                    yield process_text(sentence)

        # If extra text-data is downloaded yield that as well
        if os.path.isdir(self.EXTRA_DATA_FOLDER):
            for folder in [
                os.path.join(self.EXTRA_DATA_FOLDER, 'text'),
                os.path.join(self.EXTRA_DATA_FOLDER, 'abstracts')
            ]:
                self._print('Loading extra text from %s' % (folder))
                for filename in map(lambda x: os.path.join(folder, x), os.listdir(folder)):
                    with open(filename, 'r') as f:
                        for content in f:
                            for sentence in extract_sentences(content):
                                yield process_text(sentence)

    def dependencies_constructed(self):
        if not os.path.isfile(self.train_memmap_filename): return False
        if not os.path.isfile(self.val_memmap_filename):   return False
        if not os.path.isfile(self.test_memmap_filename):  return False
        if not os.path.isfile(self.meta_data_filename):    return False

        return True


class DataIterator:
    def __init__(self, filename, extract_func):
        self.filename = filename
        self.extract_func = extract_func

    def __iter__(self):
        with open(self.filename, 'r') as f:
            for row in f:
                yield self.extract_func(row.split(','))



class WikiCategoryCrawler:
    BASEURL = 'https://en.wikipedia.org'
    CATEGORY_CSS_SELECTOR = 'a[href^="/wiki/Category"]'
    PAGES_CSS_SELECTOR = 'a[href^="/wiki/"]'

    def __init__(self, folder, main_category_url):
        self.PAGES_FILENAME = os.path.join(folder, 'pages.csv')
        if os.path.isfile(self.PAGES_FILENAME):
            return None
        # Open pages file
        self.pages_file = open(self.PAGES_FILENAME, 'w')
        self.pages_count = 0

        # Health and fitness category page
        categories_url = self.full_url(main_category_url)

        soup = BeautifulSoup(requests.get(categories_url).text, 'html.parser')

        try:
            for anchor in soup.select(self.CATEGORY_CSS_SELECTOR):
               self.crawl_category_page(category_url=anchor.get('href'))
        except Exception:
            pass

        # Close file
        self.pages_file.close()
        print('Found %d pages' % (self.pages_count))
        print('Titles written to: %s' % (self.PAGES_FILENAME))

    @classmethod
    def read_page_titles(cls):
        with open(cls.PAGES_FILENAME, 'r') as f:
            return sorted([line.strip('\n') for line in f.readlines()])

    def full_url(self, relative_url):
        if relative_url[0] != '/':
            relative_url = '/%s' % (relative_url)
        return '%s%s' % (self.BASEURL, relative_url)

    def crawl_category_page(self, category_url):
        soup = BeautifulSoup(requests.get(self.full_url(category_url)).text, 'html.parser')

        # Go through pages links
        pages = soup.find(id='mw-pages')
        if pages is not None:
            for anchor in pages.select(self.PAGES_CSS_SELECTOR):
                page_link = anchor.get('href')
                title = page_link.split('/')[-1]
                title = parse.unquote(title)
                self.pages_file.write('%s\n' % (title))
                self.pages_count += 1
                if self.pages_count % 1000 == 0:
                    print('Found %d pages' % (self.pages_count))

        # Go through subcategories links
        subcategories = soup.find(id='mw-subcategories')
        if subcategories is not None:
            for anchor in subcategories.select(self.CATEGORY_CSS_SELECTOR):
                self.crawl_category_page(category_url=self.full_url(anchor.get('href')))



class WikiXMLParser:
    API_URL = 'http://en.wikipedia.org/w/api.php'
    RATE_LIMIT_MIN_WAIT = None
    RATE_LIMIT_LAST_CALL = None

    def __init__(self, folder, rate_limit=False):
        self.RATE_LIMIT = rate_limit
        self.ARTICLES_XML_FILENAME = os.path.join(folder, 'articles.xml')
        self.ARTICLES_CSV_FILENAME = os.path.join(folder, 'articles.csv')
        if os.path.isfile(self.ARTICLES_CSV_FILENAME):
            return None

        tree = ET.parse(self.ARTICLES_XML_FILENAME)
        root = tree.getroot()
        namespace = root.tag[0:-len('mediawiki')]

        # Precompile regular expressions
        regex_headers = re.compile(r'==(.*?)==')

        with open(self.ARTICLES_CSV_FILENAME, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['title', 'content'])
            for i, page in enumerate(root.findall('%spage' % (namespace))):
                title = page.find('%stitle' % (namespace)).text
                page_id = page.find('%sid' % (namespace)).text
                content = self.wiki_request(page_id)

                # Remove wiki markup headers
                content = regex_headers.sub(' ', content)
                content = content.replace('=', ' ')

                # Process text
                content = ' '.join(content.split())

                # Process text
                content = process_text(content)

                # Write row
                writer.writerow([title, content])

                if (i + 1) % 1000 == 0:
                    print('Downloaded and written %d pages' % (i + 1))

    def wiki_request(self, pageid):
        '''
        Make a request to the Wikipedia API using the given search parameters.
        Returns a parsed dict of the JSON response.
        '''
        query_params = {
            'action': 'query',
            'prop': 'extracts|revisions',
            'explaintext': '',
            'rvprop': 'ids',
            'format': 'json',
            'pageids': pageid
        }

        headers = {
            'User-Agent': 'wikipedia (https://github.com/goldsmith/Wikipedia/)'
        }

        if self.RATE_LIMIT and self.RATE_LIMIT_LAST_CALL and \
            self.RATE_LIMIT_LAST_CALL + self.RATE_LIMIT_MIN_WAIT > datetime.now():

            # it hasn't been long enough since the last API call
            # so wait until we're in the clear to make the request

            wait_time = (self.RATE_LIMIT_LAST_CALL + self.RATE_LIMIT_MIN_WAIT) - datetime.now()
            time.sleep(int(wait_time.total_seconds()))

        r = requests.get(self.API_URL, params=query_params, headers=headers)

        if self.RATE_LIMIT:
            self.RATE_LIMIT_LAST_CALL = datetime.now()

        json_res = r.json()
        return json_res['query']['pages'][pageid]['extract']
