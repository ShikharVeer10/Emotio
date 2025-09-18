# TODO List for Fixing test.py Code

## Step 1: Create keyword_extractor.py module
- [ ] Create the missing keyword_extractor.py file with the extract_keywords function that uses KeyBERT to extract keywords from text, considering the sentiment label.

## Step 2: Update test.py for error handling
- [ ] Add try-except block around pd.read_csv to handle FileNotFoundError if data/dataset.csv is missing.
- [ ] Ensure the code handles potential errors gracefully.

## Step 3: Test the updated code
- [ ] Run test.py to verify it executes without errors and produces the expected output.
- [ ] Check if results_with_topics.csv is generated correctly.

## Step 4: Final verification
- [ ] Confirm all imports work and no runtime errors occur.
