import os

RESULTS_DIR = 'results'

def set_standard(idx, source_path, standard_dir='standards'):
    if not os.path.exists(standard_dir):
        os.makedirs(standard_dir)
    standard_path = os.path.join(standard_dir, f'{str(idx)}.txt')
    if os.path.islink(standard_path):
        os.unlink(standard_path)
    os.symlink(source_path, standard_path)

if __name__ == '__main__':
    '''Set standard notes for all indices in results directory to full_note.txt (i.e. reference notes).'''
    for idx in os.listdir(RESULTS_DIR):
        full_note_path = os.path.join(RESULTS_DIR, str(idx), 'full_note.txt')
        if os.path.exists(full_note_path):
            set_standard(idx, full_note_path)
    
    '''
        Example usage for setting a single standard note:
    '''
    set_standard(155216, 'results/155216/full_note.txt')
