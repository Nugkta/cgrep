import pickle
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm

from cgrep import utils



def compare_subclusters(detected, known, threshold=0.5, ratio_type='known'):
    """
    Compare detected subclusters to known subclusters one-to-one.
    
    For each known subcluster, the function finds the detected subcluster that maximizes 
    the overlap ratio. If the best overlap ratio exceeds the threshold, we say that the known 
    subcluster is detected.
    
    The overlap ratio is defined as follows:
      - 'known': ratio = |known ∩ detected| / |known|
      - 'jaccard': ratio = |known ∩ detected| / |known ∪ detected|
      
    If the known clusters are provided as a list of lists, they are converted to a dictionary 
    with keys equal to the list indices.
    
    Parameters:
        detected : dict
            Dictionary of detected subclusters, where keys are detected IDs and values are lists
            (or sets) of domain strings.
        known : dict or list
            Known subclusters. If provided as a dictionary, keys are known IDs and values are lists 
            (or sets) of domain strings. If provided as a list of lists, each inner list represents 
            a known subcluster.
        threshold : float, optional (default=0.5)
            The minimum overlap ratio required to call the known subcluster as detected.
        ratio_type : str, optional (default='known')
            The type of ratio to compute. Options:
              - 'known': overlap ratio = |intersection| / |known subcluster|
              - 'jaccard': overlap ratio = |intersection| / |union|
              
    Returns:
        dict: A dictionary mapping each known subcluster ID (or index) to the detected subcluster 
              ID that best matches it if the best overlap ratio is >= threshold; otherwise, the value is None.
    """
    results = {}
    
    # Convert known clusters to a dictionary if they are provided as a list
    if isinstance(known, list):
        known_dict = {i: domains for i, domains in enumerate(known)}
    elif isinstance(known, dict):
        known_dict = known
    else:
        raise ValueError("Known clusters must be provided as a dict or a list of lists.")
    
    for known_id, known_domains in known_dict.items():
        known_set = set(known_domains)
        best_ratio = 0
        best_detected = None
        
        # Compare with each detected subcluster
        for detected_id, detected_domains in detected.items():
            detected_set = set(detected_domains)
            intersection = known_set.intersection(detected_set)
            
            if ratio_type == 'known':
                # Fraction of the known subcluster's domains that are present in the detected subcluster.
                ratio = len(intersection) / len(known_set) if known_set else 0
            elif ratio_type == 'jaccard':
                # Jaccard index: intersection divided by union.
                union = known_set.union(detected_set)
                ratio = len(intersection) / len(union) if union else 0
            else:
                raise ValueError("Unknown ratio_type. Please use 'known' or 'jaccard'.")
            
            # Keep track of the best (largest) ratio and the corresponding detected subcluster.
            if ratio > best_ratio:
                best_ratio = ratio
                best_detected = detected_id
        
        # If the best ratio is above the threshold, consider the known subcluster as detected.
        if best_ratio >= threshold:
            results[known_id] = best_detected
        else:
            results[known_id] = None
    
    return results



def compare_known_subclusters_complex(corpus_clean, model, word2idx,
                            fpath_topwords,
                            fpath_probs,
                            fpath_known_subclusters,
                            threshold_topic_presence=0.2,
                            threshold_present_score=0.5):
    """
    Process a BGC corpus to detect active subclusters and compare them with known subclusters.
    
    This function performs the following:
      - Loads the intrinsic domain composition and domain weight information for each subcluster.
      - Reads the known subclusters.
      - Iterates over each BGC in the corpus, inferring active subclusters.
      - For each active subcluster, it checks which intrinsic domains are effectively present 
        in the BGC and sums their weights.
      - If the weight sum exceeds a threshold, the subcluster is marked as present in the BGC.
      - It then compares the effective domain composition with known subclusters.
      - Finally, it computes the precision as the ratio of identified known subclusters to total known subclusters.
    
    Args:
        corpus_clean (dict): Dictionary with BGC IDs as keys and their corresponding cleaned content as values.
        model: Model instance with an infer_topic_of_doc method.
        word2idx (dict): Dictionary mapping words to indices for the model.
        fpath_topwords (str): File path for the filtered topwords pickle.
        fpath_probs (str): File path for the filtered probabilities pickle.
        fpath_known_subclusters (str): File path for the processed known subclusters.
        threshold_topic_presence (float): Threshold for inferring active subclusters in a BGC.
        threshold_present_score (float): Threshold for the summed domain weight to consider a subcluster as present.
    
    Returns:
        dict: A dictionary containing:
            - 'subcluster_presence_dic': Mapping from subcluster index to list of BGC IDs where it's present.
            - 'detected_known_sc': Mapping from detected known subcluster index to list of BGC IDs.
            - 'precision': The ratio of identified known subclusters to the total number of known subclusters.
    """
    # Load intrinsic domain compositions and topic-domain probabilities
    with open(fpath_topwords, "rb") as f:
        subc_domain_dic = pickle.load(f)
    with open(fpath_probs, "rb") as f:
        topic_domain_prob_dic = pickle.load(f)

    # Read known subclusters
    known_subclusters = utils.read_known_clusters(fpath_known_subclusters)
    
    # Prepare corpus lists and initialize dictionaries for results
    bgc_list = list(corpus_clean.keys())
    corpus_clean_list = list(corpus_clean.values())
    subcluster_presence_dic = defaultdict(list)
    detected_known_sc = defaultdict(list)

    # Iterate through the BGC database
    for idx in tqdm(range(len(corpus_clean_list)), desc='Sorting through the BGC database'):
        bgc_id = bgc_list[idx]
        bgc_current = corpus_clean[bgc_id]
        
        # Infer active subclusters in the current BGC
        active_subclusters = model.infer_topic_of_doc([bgc_current], word2idx)
        # Process each active subcluster
        print('active subclusters', active_subclusters)
        for subcluster_idx in active_subclusters:
            # print(subcluster_idx, 'the subcluster index')
            # print(corpus_clean_list[idx], 'the corpus')
            # Get intrinsic domain composition for the subcluster
            domain_comp = subc_domain_dic[subcluster_idx]
            domain_comp = np.array(domain_comp)
            # print(domain_comp, 'the domain composition')
            # check if domain 
        
            # Identify effective domains present in the BGC
            effective_domain_comp = []
            effective_domain_index = []
            # try: 
            for domain in domain_comp:
                # print('the domains of the BGC is:', corpus_clean_list[idx])
                if domain in corpus_clean_list[idx]:
                    print('domain in corpus')
                    effective_domain_comp.append(domain)
                    # print(domain)
                    # print(domain_comp)
                    effective_domain_index.append(np.where(domain_comp == domain)[0][0])
            # except:
            #     print("Error in effective domain identification", bgc_id, domain_comp, type(domain_comp), np.shape(domain_comp))
            #     break
            # Calculate the sum of weights for the effective domains
            print(effective_domain_comp, 'the effective domain composition')
            weights = []
            for j, domain in enumerate(effective_domain_comp):
                weight_i = topic_domain_prob_dic[subcluster_idx][effective_domain_index[j]]
                weights.append(weight_i)
            weight_sum = sum(weights)
            
            # If the sum of weights exceeds the threshold, mark the subcluster as present
            if weight_sum > threshold_present_score:
                subcluster_presence_dic[subcluster_idx].append(bgc_id)
            
            # Compare with known subclusters
            flag_same_as_known, detected_subclusters_idx = utils.identify_known_subclusters(effective_domain_comp, known_subclusters)
            if flag_same_as_known:
                detected_known_sc[detected_subclusters_idx].append(bgc_id)

    # Compute precision
    identified_subclusters_num = len(detected_known_sc)
    total_subclusters_num = len(known_subclusters)
    precision = identified_subclusters_num / total_subclusters_num if total_subclusters_num > 0 else 0

    return {
        'subcluster_presence_dic': subcluster_presence_dic,
        'detected_known_sc': detected_known_sc,
        'precision': precision
    }



