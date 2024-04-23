//
//  rpfbwt_algorithm.hpp
//

#ifndef rpfbwt_algorithm_hpp
#define rpfbwt_algorithm_hpp

#include <filesystem>
#include <sys/stat.h>
#include <omp.h>

#include <pfp/pfp.hpp>

#undef max
#include <rle/rle_string.hpp>

#include <sampled_lcp_support.hpp>
// #include <p_inverted_list_support.hpp>
#include <v_table_support.hpp>
#include <end_arrays_support.hpp>

namespace rpfbwt
{

template <typename dict_l1_data_type, typename parse_int_type = uint32_t>
class rpfbwt_algo
{
public:
    
    class l2_colex_comp
    {
    private:
        
        const pfpds::dictionary<dict_l1_data_type>& l1_d;
        parse_int_type int_shift;
    
    public:
        
        l2_colex_comp(const pfpds::dictionary<dict_l1_data_type>& l1_d_ref, parse_int_type shift)
        : l1_d(l1_d_ref) , int_shift(shift) {}
        
        bool operator()(parse_int_type l, parse_int_type r) const
        {
            assert(l != r);
            if (l < int_shift) { return true; }
            if (r < int_shift) { return false; }
            return l1_d.colex_id[l - int_shift - 1] < l1_d.colex_id[r - int_shift - 1];
        }
    };
    
private:
    
    pfpds::long_type int_shift = 10;
    std::string l1_prefix;
    std::string out_rle_name;
    
    std::less<dict_l1_data_type> l1_d_comp;
    const pfpds::dictionary<dict_l1_data_type>& l1_d;

    std::map<int, std::tuple<pfpds::long_type, pfpds::long_type, pfpds::long_type, pfpds::long_type>> chunk_border_SA_info;

    l2_colex_comp l2_comp;
    const pfpds::dictionary<parse_int_type, l2_colex_comp>& l2_d;
    const pfpds::parse& l2_p;
    const pfpds::pf_parsing<parse_int_type, l2_colex_comp, pfpds::pfp_wt_sdsl>& l2_pfp;
    v_table_support<parse_int_type, l2_colex_comp, pfpds::pfp_wt_sdsl> l2_pfp_v_table; // (row in which that char appears, number of times per row)
    recursive_slcp_support<dict_l1_data_type, l2_colex_comp, parse_int_type> slcp_support;
    end_arrays_support<dict_l1_data_type, l2_colex_comp, parse_int_type> E_arrays;
    pfpds::long_type l1_n = 0;

    static constexpr pfpds::long_type default_num_of_chunks = 1;
    std::vector<std::tuple<pfpds::long_type, pfpds::long_type, pfpds::long_type, pfpds::long_type>> chunks;
    
    rle::RLEString::RLEncoderMerger rle_chunks;
    
    // Computes ranges for parallel computation
    // suffix start, suffix end, this_left, this_row
    void
    compute_chunks(pfpds::long_type num_of_chunks)
    {
        spdlog::info("Computing chunks for parallel execution. Total input size: {}. Requested chunks: {}", l1_n, num_of_chunks);
        
        pfpds::long_type chunk_size = (num_of_chunks > 1) ? (l1_n / (num_of_chunks)) : l1_n + 1;
        
        // Go through the suffixes of D and compute chunks
        pfpds::long_type i = 1; // This should be safe since the first entry of sa is always the dollar sign used to compute the sa
        
        pfpds::long_type l_left  = 0;
        pfpds::long_type l_right = 0;
        pfpds::long_type chunk_suffix_start = i;
        pfpds::long_type chunk_start = l_left;
        pfpds::long_type chunk_row_start = 0;
        pfpds::long_type table_row = 0;
        while (i < l1_d.saD.size())
        {
            auto sn = l1_d.saD[i];
            // Check if the suffix has length at least w and is not the complete phrase.
            auto phrase = l1_d.daD[i] + 1;
            assert(phrase > 0 and phrase < (l1_d.n_phrases() + 1)); // + 1 because daD is 0-based
            pfpds::long_type suffix_length = l1_d.select_b_d(l1_d.rank_b_d(sn + 1) + 1) - sn - 1;
            if (l1_d.b_d[sn] or suffix_length < l1_d.w) // skip full phrases or suffixes shorter than w
            {
                ++i; // Skip
            }
            else
            {
                i++;
                l_right += E_arrays.l1_freq(phrase) - 1;
                
                if (i < l1_d.saD.size())
                {
                    auto new_sn = l1_d.saD[i];
                    auto new_phrase = l1_d.daD[i] + 1;
                    assert(new_phrase > 0 and new_phrase < (l1_d.n_phrases() + 1)); // + 1 because daD is 0-based
                    pfpds::long_type new_suffix_length = l1_d.select_b_d(l1_d.rank_b_d(new_sn + 1) + 1) - new_sn - 1;
                    
                    while (i < l1_d.saD.size() && (l1_d.lcpD[i] >= suffix_length) and (suffix_length == new_suffix_length))
                    {
                        ++i;
                        l_right += E_arrays.l1_freq(new_phrase);
                        
                        if (i < l1_d.saD.size())
                        {
                            new_sn = l1_d.saD[i];
                            new_phrase = l1_d.daD[i] + 1;
                            assert(new_phrase > 0 and new_phrase < (l1_d.n_phrases() + 1)); // + 1 because daD is 0-based
                            new_suffix_length = l1_d.select_b_d(l1_d.rank_b_d(new_sn + 1) + 1) - new_sn - 1;
                        }
                    }
                    
                }
                
                if (l_right - chunk_start > chunk_size)
                {
                    // store SA_d range for this chunk
                    chunks.emplace_back(chunk_suffix_start, i, chunk_start, chunk_row_start);
                    
                    // prepare for next iteration
                    chunk_suffix_start = i;
                    chunk_start = l_right + 1;
                    chunk_row_start = table_row + 1;
                }
                
                l_left = l_right + 1;
                l_right = l_left;
                table_row++;
            }
        }
        
        // add last_chunk
        if ((chunks.empty()) or (std::get<1>(chunks.back()) < i))
        {
            // store SA_d range for last chunk
            chunks.emplace_back(chunk_suffix_start, i, chunk_start, chunk_row_start);
        }
        
        assert(chunks.size() == num_of_chunks);
        spdlog::info("BWT ranges divided in {} chunks.", chunks.size());
    }
    
public:
    // std::tuple<pfpds::long_type, pfpds::long_type, pfpds::long_type, pfpds::long_type> l1_sa_values_chunk(const std::tuple<pfpds::long_type, pfpds::long_type, pfpds::long_type, pfpds::long_type>& chunk,
    //                         const rle::bitvector& run_heads_bitvector,
    //                         const std::string& tmp_file_name);
    rpfbwt_algo(
    std::string prefix,
    const pfpds::dictionary<dict_l1_data_type>& l1_d_r,
    const pfpds::pf_parsing<parse_int_type, l2_colex_comp, pfpds::pfp_wt_sdsl>& l2_pfp_r,
    pfpds::long_type pfp_integer_shift,
    pfpds::long_type bwt_chunks = default_num_of_chunks)
    :
    int_shift(pfp_integer_shift),
    l1_d(l1_d_r),
    l1_prefix(prefix),
    out_rle_name(l1_prefix + ".rlebwt"),
    l2_comp(l1_d, int_shift),
    l2_d(l2_pfp_r.dict),
    l2_p(l2_pfp_r.pars),
    l2_pfp(l2_pfp_r),
    l2_pfp_v_table(l2_pfp),
    E_arrays(l1_d, l2_d, l2_p, int_shift),
    slcp_support(l1_d, l2_d, l2_p, int_shift),
    rle_chunks(out_rle_name, bwt_chunks)
    {
        assert(l1_d.saD_flag);
        assert(l1_d.daD_flag);
        assert(l1_d.lcpD_flag);
        assert(l1_d.colex_id_flag);
        assert(l2_pfp.bwt_P_ilist_flag);
        assert(l2_pfp.dict.colex_id_flag);
        assert(l2_pfp.pars.saP_flag);
        
        this->l1_n = E_arrays.l1_length();
        compute_chunks(bwt_chunks);
    }
    
    void
    l1_bwt_chunk(
    const std::tuple<pfpds::long_type, pfpds::long_type, pfpds::long_type, pfpds::long_type>& chunk,
    rle::RLEString::RLEncoder& rle_out)
    {
        pfpds::long_type i = std::get<0>(chunk); //chunk suffix start
        
        dict_l1_data_type prev_char = 0;
    
        pfpds::long_type l_left  = std::get<2>(chunk); //chunk start
        pfpds::long_type l_right = l_left;
        pfpds::long_type easy_chars = 0;
        pfpds::long_type hard_easy_chars = 0;
        pfpds::long_type hard_hard_chars = 0;
        pfpds::long_type row = std::get<3>(chunk); //chunk row
        while (i < std::get<1>(chunk))//chunk suffix end
        {
            pfpds::long_type sn = l1_d.saD[i]; //Suffix index
            // Check if the suffix has length at least w and is not the complete phrase.
            pfpds::long_type phrase = l1_d.daD[i] + 1; //Suffix phrase id
            assert(phrase > 0 and phrase < (l1_d.n_phrases() + 1)); // + 1 because daD is 0-based
            pfpds::long_type suffix_length = l1_d.select_b_d(l1_d.rank_b_d(sn + 1) + 1) - sn - 1; //select rank of phrase that is after current suffix in dictionary.
            if (l1_d.b_d[sn] || suffix_length < l1_d.w) // skip full phrases or suffixes shorter than w
            {
                ++i; // Skip
            }
            else
            {
                std::set<dict_l1_data_type> chars;
                chars.insert(l1_d.d[((sn + l1_d.d.size() - 1) % l1_d.d.size())]); //character before suffix
    
                std::set<parse_int_type> pids;
                pids.insert(phrase); //store phrase #
                
                i++;
                l_right += E_arrays.l1_freq(phrase) - 1;
            
                if (i < l1_d.saD.size())
                {
                    pfpds::long_type new_sn = l1_d.saD[i];
                    pfpds::long_type new_phrase = l1_d.daD[i] + 1;
                    assert(new_phrase > 0 and new_phrase < (l1_d.n_phrases() + 1)); // + 1 because daD is 0-based
                    pfpds::long_type new_suffix_length = l1_d.select_b_d(l1_d.rank_b_d(new_sn + 1) + 1) - new_sn - 1;
                
                    while (i < l1_d.saD.size() && (l1_d.lcpD[i] >= suffix_length) and (suffix_length == new_suffix_length))
                    {
                        chars.insert(l1_d.d[((new_sn + l1_d.d.size() - 1) % l1_d.d.size())]);
                        pids.insert(new_phrase);
                        ++i;
                    
                        l_right += E_arrays.l1_freq(new_phrase);
                    
                        if (i < l1_d.saD.size())
                        {
                            new_sn = l1_d.saD[i];
                            new_phrase = l1_d.daD[i] + 1;
                            assert(new_phrase > 0 and new_phrase < (l1_d.n_phrases() + 1)); // + 1 because daD is 0-based
                            new_suffix_length = l1_d.select_b_d(l1_d.rank_b_d(new_sn + 1) + 1) - new_sn - 1;
                        }
                    }
                
                }//Suffixes are of a document made from the dictionary, which means that proper phrase suffixes are not without context.
                //This means that the same proper phrase suffixes cannot be collapsed, but they are guaranteed together.
                //This will find all characters that precede the same proper phrase suffix.
                
                if (chars.size() == 1) // easy-easy suffixes
                {
                    // easy suffixes
                    rle_out(*(chars.begin()), (l_right - l_left) + 1);
                    easy_chars += (l_right - l_left) + 1;
                }
                else // easy-hard and hard-hard suffixes
                {
                    assert(l_right - l_left > 0);
                    pfpds::long_type hard_chars_before = hard_easy_chars + hard_hard_chars;
                    
                    // shorthands
                    typedef std::reference_wrapper<const std::vector<std::pair<pfpds::long_type, pfpds::long_type>>> ve_t; // (row in which that char appears, number of times per row)
                    typedef pfpds::long_type ve_pq_t; // for the priority queue the row is enough, the other ingo can be retrieved from ve_t
                    typedef std::pair<ve_pq_t, std::pair<pfpds::long_type, pfpds::long_type>> pq_t; // .first : row, this element is the .second.second-th element of the .second.first array
                    
                    // go in second level and iterate through list of positions for the meta-phrases we are interested into
                    std::vector<ve_t> v;
                    std::vector<parse_int_type> pids_v;
                    for (const auto& pid : pids)
                    {
                        parse_int_type adj_pid = pid + int_shift; //this gives us the meta-character from the phrase id
                        v.push_back(std::cref(l2_pfp_v_table[adj_pid])); //get the v-table information for the meta-phrases of interest
                        pids_v.push_back(adj_pid);
                    }

                    // make a priority queue and add elements to it
                    std::priority_queue<pq_t, std::vector<pq_t>, std::greater<pq_t>> pq;
                    //v[vi].get() is vector of all L2 suffixes that have a specific meta-character before it. 
                    for (pfpds::long_type vi = 0; vi < v.size(); vi++) { pq.push({ v[vi].get()[0].first, { vi, 0 } }); }
                    //Priority queue based on row/suffix ordering (all occurrences of the same suffix occur in sequence). 
                    // Stores row (suffix) that has this metacharacter before it.
                    //Also stores which metacharacter and which occurrence of this metacharacter it is.
                    while (not pq.empty())
                    {
                        // get all chars from this row entry at l2
                        std::vector<pq_t> from_same_l2_suffix;
                        auto first_suffix = pq.top();
                        while ((not pq.empty()) and (pq.top().first == first_suffix.first))
                        {
                            auto curr = pq.top(); pq.pop();
                            from_same_l2_suffix.push_back(curr);

                            pfpds::long_type arr_i_c = curr.second.first;  // ith array
                            pfpds::long_type arr_x_c = curr.second.second; // index in i-th array
                            if (arr_x_c + 1 < v[arr_i_c].get().size())
                            {
                                pq.push({ v[arr_i_c].get()[arr_x_c + 1].first, { arr_i_c, arr_x_c + 1 } });
                            }
                        }
                        //Adds all metacharacters of interest at this row to from_same_l2_suffix

                        if (from_same_l2_suffix.empty())
                        {
                            spdlog::error("Something went wrong.");
                        }
                        else if (from_same_l2_suffix.size() == 1)
                        {
                            // hard-easy suffix, we can fill in the character in pid
                            auto pq_entry = from_same_l2_suffix[0];
                            parse_int_type adj_pid = pids_v[pq_entry.second.first];
                            pfpds::long_type freq = v[pq_entry.second.first].get()[pq_entry.second.second].second;
                            dict_l1_data_type c = l1_d.d[l1_d.select_b_d(adj_pid - int_shift + 1) - (suffix_length + 2)]; // end of next phrases
                            
                            rle_out(c, freq);
                            hard_easy_chars += freq;
                        }
                        else
                        {
                            // hard-hard suffix, need to look at the grid in L2
                            pfpds::long_type curr_l2_row = from_same_l2_suffix[0].first;
                            auto l2_M_entry = l2_pfp.M[curr_l2_row];
                            
                            // get inverted lists of corresponding phrases
                            std::vector<pfpds::long_type> ilists_idxs;
                            std::vector<dict_l1_data_type> ilist_corresponding_chars;
                            for (pfpds::long_type c_it = l2_M_entry.left; c_it <= l2_M_entry.right; c_it ++)
                            {
                                // check if we need that pid
                                pfpds::long_type l2_pid = l2_pfp.dict.colex_id[c_it] + 1;
                                parse_int_type adj_l1_pid = l2_pfp.dict.d[l2_pfp.dict.select_b_d(l2_pid + 1) - (l2_M_entry.len + 2)];
                                assert(adj_l1_pid >= int_shift);
                                parse_int_type l1_pid = adj_l1_pid - int_shift;
    
                                if (pids.find(l1_pid) != pids.end())
                                {
                                    ilists_idxs.push_back(l2_pid);
                                    dict_l1_data_type c = l1_d.d[l1_d.select_b_d(l1_pid + 1) - (suffix_length + 2)];
                                    ilist_corresponding_chars.push_back(c);
                                }
                            }
     
                            // make a priority queue from the inverted lists
                            typedef std::pair<pfpds::long_type, std::pair<pfpds::long_type, pfpds::long_type>> ilist_pq_t;
                            std::priority_queue<ilist_pq_t, std::vector<ilist_pq_t>, std::greater<ilist_pq_t>> ilist_pq;
                            for (pfpds::long_type il_i = 0; il_i < ilists_idxs.size(); il_i++)
                            {
                                ilist_pq.push({ l2_pfp.bwt_p_ilist[ilists_idxs[il_i]][0], { il_i, 0 } });
                                //For each meta-phrase that is preceeded by a metacharacter of interest, we store its first occurrence in the BWT of P_P
                            }
                            
                            // now pop elements from the priority queue and write out the corresponding character
                            while (not ilist_pq.empty())
                            {
                                auto curr = ilist_pq.top(); ilist_pq.pop();
                                
                                // output corresponding char
                                dict_l1_data_type c = ilist_corresponding_chars[curr.second.first];

                                rle_out(c, 1);
                                hard_hard_chars += 1;
                                
                                // keep iterating
                                pfpds::long_type arr_i_c_il = curr.second.first;  // ith array
                                pfpds::long_type arr_x_c_il = curr.second.second; // index in i-th array
                                if (arr_x_c_il + 1 < l2_pfp.bwt_p_ilist[ilists_idxs[arr_i_c_il]].size())
                                {
                                    ilist_pq.push({ l2_pfp.bwt_p_ilist[ilists_idxs[arr_i_c_il]][arr_x_c_il + 1], { arr_i_c_il, arr_x_c_il + 1 } });
                                }
                                //Add its next occurrence to the priority queue
                            }
                            //Do this for every meta-phrase that contains a metacharacter of interest.
                            //i.e. the meta-phrase has a proper meta-phrase suffix that has a metacharacter of interest before it.
                            //Note these phrases all contain the same meta-phrase suffix of interest.
                        }
                    }
                    //We do this for all proper meta-phrase suffixes that have a metacharacter of interest before it.
                    
                    // check that we covered the range we were working on
                    assert( ((hard_easy_chars + hard_hard_chars) - hard_chars_before) == ((l_right - l_left) + 1) );
                }
                
                l_left = l_right + 1;
                l_right = l_left;
                row++;
            }
            //We move on to the next proper phrase suffix.
        }
    }
    
    //------------------------------------------------------------
    
    std::tuple<pfpds::long_type, pfpds::long_type, pfpds::long_type, pfpds::long_type>
    l1_sa_values_chunk(
    const std::tuple<pfpds::long_type, pfpds::long_type, pfpds::long_type, pfpds::long_type>& chunk,
    const rle::bitvector& run_heads_bitvector,
    const std::string& sa_tmp_file_name, const std::string& lcp_tmp_file_name)
    {
        pfpds::long_type skipped_portion = 0;
        pfpds::long_type processed_portion = 0;
        //text index, l1_pid, l2_pid, l1_suffix_length, l2_suffix start
        pfpds::long_type i = std::get<0>(chunk); //chunk suffix start
        
        std::tuple<pfpds::long_type, pfpds::long_type, pfpds::long_type, pfpds::long_type> prev_SA_info = std::make_tuple(-1, -1, -1, -1);

        std::ofstream out_sa_tmp_fstream(sa_tmp_file_name, std::ios::out | std::ios::binary);
        std::ofstream out_lcp_tmp_fstream(lcp_tmp_file_name, std::ios::out | std::ios::binary);
    
        pfpds::long_type sa_pos_iterator = 0;
        
        pfpds::long_type l_left  = std::get<2>(chunk); //chunk start
        pfpds::long_type l_right = l_left;
        pfpds::long_type easy_chars = 0;
        pfpds::long_type hard_easy_chars = 0;
        pfpds::long_type hard_hard_chars = 0;
        pfpds::long_type row = std::get<3>(chunk); //chunk row
        while (i < std::get<1>(chunk)) //last index in chunk
        {
            auto sn = l1_d.saD[i];
            // Check if the suffix has length at least w and is not the complete phrase.
            auto phrase = l1_d.daD[i] + 1;
            assert(phrase > 0 and phrase < (l1_d.n_phrases() + 1)); // + 1 because daD is 0-based
            pfpds::long_type suffix_length = l1_d.select_b_d(l1_d.rank_b_d(sn + 1) + 1) - sn - 1;
            if (l1_d.b_d[sn] or suffix_length < l1_d.w) // skip full phrases or suffixes shorter than w
            {
                ++i; // Skip
            }
            else
            {
                std::set<dict_l1_data_type> chars;
                chars.insert(l1_d.d[((sn + l1_d.d.size() - 1) % l1_d.d.size())]);
            
                std::set<parse_int_type> pids;
                pids.insert(phrase);
            
                i++;
                l_right += E_arrays.l1_freq(phrase) - 1;
            
                if (i < l1_d.saD.size())
                {
                    auto new_sn = l1_d.saD[i];
                    auto new_phrase = l1_d.daD[i] + 1;
                    assert(new_phrase > 0 and new_phrase < (l1_d.n_phrases() + 1)); // + 1 because daD is 0-based
                    pfpds::long_type new_suffix_length = l1_d.select_b_d(l1_d.rank_b_d(new_sn + 1) + 1) - new_sn - 1;
                
                    while (i < l1_d.saD.size() && (l1_d.lcpD[i] >= suffix_length) && (suffix_length == new_suffix_length))
                    {
                        chars.insert(l1_d.d[((new_sn + l1_d.d.size() - 1) % l1_d.d.size())]);
                        pids.insert(new_phrase);
                        ++i;
                    
                        l_right += E_arrays.l1_freq(new_phrase);
                    
                        if (i < l1_d.saD.size())
                        {
                            new_sn = l1_d.saD[i];
                            new_phrase = l1_d.daD[i] + 1;
                            assert(new_phrase > 0 and new_phrase < (l1_d.n_phrases() + 1)); // + 1 because daD is 0-based
                            new_suffix_length = l1_d.select_b_d(l1_d.rank_b_d(new_sn + 1) + 1) - new_sn - 1;
                        }
                    }
                
                }
                //Find all other occurrences of the current proper phrase suffix + the phrase ids of the phrases that contain them
                //+ the characters that precede the suffix

                pfpds::long_type run_heads_in_range = 0;
                if (run_heads_bitvector[l_left]) { run_heads_in_range = 1; } // at least
                else if(l_right != run_heads_bitvector.size()-1) { run_heads_in_range = run_heads_bitvector.rank(l_right + 2) - run_heads_bitvector.rank(l_left); }
                else {run_heads_in_range = run_heads_bitvector.rank(l_right + 1) - run_heads_bitvector.rank(l_left);} // include the extremes
                
                if (run_heads_in_range != 0) // removing skipping to be able to do LCP calculation
                {
                    pfpds::long_type sa_values_c = 0; // number of sa values computed in this range
                    
                    // shorthands
                    typedef std::reference_wrapper<const std::vector<std::pair<pfpds::long_type, pfpds::long_type>>> ve_t; // (row in which that char appears, number of times per row)
                    typedef pfpds::long_type ve_pq_t; // for the priority queue the row is enough, the other ingo can be retrieved from ve_t
                    typedef std::pair<ve_pq_t, std::pair<pfpds::long_type, pfpds::long_type>> pq_t; // .first : row, this element is the .second.second-th element of the .second.first array
    
                    // go in second level and iterate through list of positions for the meta-phrases we are interested into
                    std::vector<ve_t> v;
                    std::vector<parse_int_type> pids_v;
                    for (const auto& pid : pids)
                    {
                        parse_int_type adj_pid = pid + int_shift;
                        v.push_back(std::cref(l2_pfp_v_table[adj_pid]));
                        pids_v.push_back(adj_pid);
                    }
                    //For each meta-character of interest, get its list of where it occurs in the suffix array of D_P
    
                    // make a priority queue and add elements to it
                    std::priority_queue<pq_t, std::vector<pq_t>, std::greater<pq_t>> pq;
                    for (pfpds::long_type vi = 0; vi < v.size(); vi++) { pq.push({ v[vi].get()[0].first, { vi, 0 } }); }
                    //Priority Queue based on the row number in the SA of D_P. v[vi].get()[0].first is the row number.
                    while (not pq.empty())
                    {
                        // get all chars from this row entry at l2
                        std::vector<pq_t> from_same_l2_suffix;
                        auto first_suffix = pq.top();
                        while ((not pq.empty()) and (pq.top().first == first_suffix.first))
                        {
                            auto curr = pq.top(); pq.pop();
                            from_same_l2_suffix.push_back(curr);
            
                            pfpds::long_type arr_i_c = curr.second.first;  // ith array
                            pfpds::long_type arr_x_c = curr.second.second; // index in i-th array
                            if (arr_x_c + 1 < v[arr_i_c].get().size())
                            {
                                pq.push({ v[arr_i_c].get()[arr_x_c + 1].first, { arr_i_c, arr_x_c + 1 } });
                            }
                        }
        
                        // hard-hard suffix, need to look at the grid in L2
                        pfpds::long_type curr_l2_row = from_same_l2_suffix[0].first;
                        auto& l2_M_entry = l2_pfp.M[curr_l2_row];
        
                        // get inverted lists of corresponding phrases
                        std::vector<pfpds::long_type> ilists_idxs;
                        std::vector<std::reference_wrapper<const std::vector<pfpds::long_type>>> ilists_e_arrays;
                        std::vector<pfpds::long_type> ilist_corresponding_sa_expanded_values;
                        std::vector<pfpds::long_type> ilist_suffix_starts;
                        for (pfpds::long_type c_it = l2_M_entry.left; c_it <= l2_M_entry.right; c_it++)
                        {
                            // check if we need that pid
                            pfpds::long_type l2_pid = l2_pfp.dict.colex_id[c_it] + 1;
                            parse_int_type adj_l1_pid = l2_pfp.dict.d[l2_pfp.dict.select_b_d(l2_pid + 1) - (l2_M_entry.len + 2)];
                            assert(adj_l1_pid >= int_shift);
                            parse_int_type l1_pid = adj_l1_pid - int_shift;
    
                            if (pids.find(l1_pid) != pids.end())
                            {
                                ilists_idxs.push_back(l2_pid);
                                ilists_e_arrays.push_back(std::ref(E_arrays[l2_pid - 1]));
                                
                                // get the length of the current l2_suffix by expanding phrases
                                pfpds::long_type l2_suff_start = l2_pfp.dict.select_b_d(l2_pid + 1) - 2 - (l2_M_entry.len - 1);
                                pfpds::long_type l2_suff_end = l2_pfp.dict.select_b_d(l2_pid + 1) - 2 - (l2_pfp.w - 1) - 1;
                                pfpds::long_type sa_r = 0;
                                for (pfpds::long_type p_it_b = l2_suff_start; p_it_b <= l2_suff_end; p_it_b++)
                                {
                                    pfpds::long_type adj_pid = l2_pfp.dict.d[p_it_b] - int_shift;
                                    sa_r += l1_d.lengths[adj_pid - 1] - l1_d.w; // l1_d.select_b_d(adj_pid + 1) - l1_d.select_b_d(adj_pid) - 1 - l1_d.w;
                                }
                                ilist_corresponding_sa_expanded_values.push_back(sa_r); // the sum of the lengsh of each l1 pid in the l2 phrase suffix
                                ilist_suffix_starts.push_back(l2_suff_start);
                            }
                        }
        
                        // make a priority queue from the inverted lists
                        typedef std::pair<pfpds::long_type, std::pair<pfpds::long_type, pfpds::long_type>> ilist_pq_t;
                        std::priority_queue<ilist_pq_t, std::vector<ilist_pq_t>, std::greater<ilist_pq_t>> ilist_pq;
                        for (pfpds::long_type il_i = 0; il_i < ilists_idxs.size(); il_i++)
                        {
                            ilist_pq.push({ l2_pfp.bwt_p_ilist[ilists_idxs[il_i]][0], { il_i, 0 } });
                        }
                        // from bwt_p_ilist we can grab all positions in the BWT of P of specific metaphrases, determined by ilists_idxs
                        //PQ based on the ordering of each l2 phrase in the bwt of P. il_i is how to find the corresponding phrase id and end arrays value.
                        // now pop elements from the priority queue and write out the corresponding character
                        while (not ilist_pq.empty())
                        {
                            auto curr = ilist_pq.top(); ilist_pq.pop();
                            // top is the next phrase in the bwt order
                            // compute corresponding sa value
                            pfpds::long_type phrase_end = ilists_e_arrays[curr.second.first].get()[curr.second.second]; //get first ending position of this phrase in original text.
                            pfpds::long_type out_sa_value = phrase_end - (ilist_corresponding_sa_expanded_values[curr.second.first]); //phrase end minus l2 suffix length (in text) = index in original text
            
                            // adjust sa value for circular representation
                            if (out_sa_value > suffix_length) { out_sa_value -= suffix_length; }
                            else { out_sa_value = (l1_n + out_sa_value - suffix_length) % l1_n; }
                            //Subtract the l1 suffix length to get original index in text
                            

                            pfpds::long_type l2_pid = ilists_idxs[curr.second.first];
                            parse_int_type l1_pid = l2_pfp.dict.d[l2_pfp.dict.select_b_d(l2_pid + 1) - (l2_M_entry.len + 2)] - int_shift;

                            pfpds::long_type l1_d_index = l1_d.select_b_d(l1_pid+1) - suffix_length - 1;
                            pfpds::long_type l2_d_index = ilist_suffix_starts[curr.second.first];
                            pfpds::long_type l2_p_index = l2_p.saP[l2_pfp.bwt_p_ilist[l2_pid][curr.second.second] + 1];
                            pfpds::long_type sa_r = ilist_corresponding_sa_expanded_values[curr.second.first];

                            std::tuple<pfpds::long_type, pfpds::long_type, pfpds::long_type, pfpds::long_type> curr_SA_info = std::make_tuple(out_sa_value, l1_d_index, l2_d_index, l2_p_index);

                            // output if run head
                            if (run_heads_bitvector[l_left + sa_values_c])
                            {
                                pfpds::long_type out_lcp_value = calculate_lcp(curr_SA_info, prev_SA_info, suffix_length, sa_r);
                                printf("SA: %lu LCP: %lu\n", out_sa_value, out_lcp_value);
                                out_sa_tmp_fstream.write((char*)&out_sa_value, sizeof(pfpds::long_type));
                                out_lcp_tmp_fstream.write((char*)&out_lcp_value, sizeof(pfpds::long_type));
                                sa_pos_iterator += 1;
                            }
                            prev_SA_info = curr_SA_info;
            
                            // keep iterating
                            pfpds::long_type arr_i_c_il = curr.second.first;  // ith array
                            pfpds::long_type arr_x_c_il = curr.second.second; // index in i-th array
                            if (arr_x_c_il + 1 < l2_pfp.bwt_p_ilist[ilists_idxs[arr_i_c_il]].size())
                            {
                                ilist_pq.push({ l2_pfp.bwt_p_ilist[ilists_idxs[arr_i_c_il]][arr_x_c_il + 1], { arr_i_c_il, arr_x_c_il + 1 } });
                                //Making sure to visit all occurrences of this L2 suffix in BWT_P order
                            }
                            
                            sa_values_c++;
                        }
                    }
                    processed_portion += l_right - l_left;
                }
                else // we skipped this portions
                {
                    skipped_portion += l_right - l_left;
                }
            
                l_left = l_right + 1;
                l_right = l_left;
                row++;
            }
        }
        spdlog::info("Processed: {} Skipped: {}", processed_portion, skipped_portion);
    
        // close tmp file stream
        out_sa_tmp_fstream.close();
        out_sa_tmp_fstream.close();
        return prev_SA_info;
    }
    
    pfpds::long_type calculate_lcp(std::tuple<pfpds::long_type, pfpds::long_type, pfpds::long_type, pfpds::long_type> curr, 
    std::tuple<pfpds::long_type, pfpds::long_type, pfpds::long_type, pfpds::long_type> prev, 
    pfpds::long_type suffix_length, pfpds::long_type sa_r)
    {
        bool l1_phrase_boundary = false;
        bool l2_phrase_boundary = false;
        auto min = 0;
        auto max = 0;
        pfpds::long_type out_lcp_value = 0;
        if(std::get<0>(prev) != -1)
        {
            if(std::get<1>(curr) == std::get<1>(prev))
            {
                out_lcp_value += suffix_length - l1_d.w;
                l1_phrase_boundary = true;
            }
            else
            {
                auto l1_sa_prev = l1_d.isaD[std::get<1>(prev)];
                auto l1_sa_curr = l1_d.isaD[std::get<1>(curr)];
                auto min = std::min(l1_sa_prev, l1_sa_curr)+1;
                auto max = std::max(l1_sa_prev, l1_sa_curr);                                  
                out_lcp_value += l1_d.lcpD[l1_d.rmq_lcp_D(min, max)];
                if(l1_d.d[std::get<1>(curr) + out_lcp_value] == EndOfWord)
                {
                    out_lcp_value -= l1_d.w;
                    l1_phrase_boundary = true;
                }
            }

            if(l1_phrase_boundary) // we fully match proper phrase suffix and need to check for matches beyond
            {

                if(std::get<2>(curr) == std::get<2>(prev))
                {
                    out_lcp_value += sa_r;
                    l2_phrase_boundary = true;
                }
                else
                {
                    auto l2_sa_prev = l2_d.isaD[std::get<2>(prev)];
                    auto l2_sa_curr = l2_d.isaD[std::get<2>(curr)];
                    min = std::min(l2_sa_prev, l2_sa_curr)+1;
                    max = std::max(l2_sa_prev, l2_sa_curr);
                    pfpds::long_type meta_matches = l2_d.lcpD[l2_d.rmq_lcp_D(min, max)];
                    if(l2_d.d[std::get<2>(curr) + meta_matches] == EndOfWord or l2_d.d[std::get<2>(prev) + meta_matches] == EndOfWord)
                    {
                        out_lcp_value += sa_r;
                        l2_phrase_boundary = true;
                    }
                    else
                    {
                        out_lcp_value += slcp_support.extended_lcp_l2_D(slcp_support.rmq_eLCP_l2_D(min, max));
                    }
                }

                if(l2_phrase_boundary)
                {
                    auto l2_saP_curr = l2_p.isaP[std::get<3>(curr)];
                    auto l2_saP_prev = l2_p.isaP[std::get<3>(prev)];
                    min = std::min(l2_saP_prev, l2_saP_curr) + 1;
                    max = std::max(l2_saP_prev, l2_saP_curr);
                    out_lcp_value += slcp_support.extended_lcp_l2_P(slcp_support.rmq_eLCP_l2_P(min, max));
                }
            }
        }
        return out_lcp_value;
    }


    //------------------------------------------------------------
    
    void l1_rlebwt(pfpds::long_type threads = 1)
    {
        // Set threads accordingly to configuration
        omp_set_num_threads(threads);
    
        // ----------
        // Compute run length encoded bwt and run heads positions
        #pragma omp parallel for schedule(dynamic) default(none)
        for (pfpds::long_type i = 0; i < chunks.size(); i++)
        {
            l1_bwt_chunk(chunks[i], rle_chunks.get_encoder(i));
            spdlog::info("Chunk {}/{} completed", i + 1, chunks.size());
        }
        rle_chunks.close();
    }
    
    
    //------------------------------------------------------------
    
    void l1_refined_rindex(pfpds::long_type threads = 1)
    { 
        // Set threads accordingly to configuration
        omp_set_num_threads(threads);
        
        // ----------
        // Compute run length encoded bwt and run heads positions
        #pragma omp parallel for schedule(dynamic) default(none)
        for (pfpds::long_type i = 0; i < chunks.size(); i++)
        {
            l1_bwt_chunk(chunks[i], rle_chunks.get_encoder(i));
            spdlog::info("Chunk {}/{} completed", i + 1, chunks.size());
        }
        rle_chunks.close();
        // ----------
        // Get bitvector marking run heads from the rle bwt
        rle::RLEString::RLEDecoder rle_decoder(out_rle_name);
        sdsl::sd_vector_builder run_heads_bv_builder(rle_decoder.metadata.size, rle_decoder.metadata.runs);
        pfpds::long_type runs_bv_it = 0;
    
        while (not rle_decoder.end())
        {
            rle::RunType run = rle_decoder.next();
            dict_l1_data_type c = rle::RunTraits::charachter(run);
            pfpds::long_type len = rle::RunTraits::length(run);
        
            if (len != 0)
            { run_heads_bv_builder.set(runs_bv_it); runs_bv_it += len; }
        }
    
        rle::bitvector run_heads_bv = rle::bitvector(run_heads_bv_builder);
    
        // ----------
        // Compute sa values at run heads
        std::vector<std::string> sa_chunks_tmp_files;
        std::vector<std::string> lcp_chunks_tmp_files;
        for (pfpds::long_type i = 0; i < chunks.size(); i++) 
        { 
            sa_chunks_tmp_files.push_back(rle::TempFile::getName("sa_chunk")); 
            lcp_chunks_tmp_files.push_back(rle::TempFile::getName("lcp_chunk"));
        }
        
        #pragma omp parallel for schedule(dynamic) default(none) shared(run_heads_bv, sa_chunks_tmp_files, lcp_chunks_tmp_files)
        for (pfpds::long_type i = 0; i < chunks.size(); i++)
        {
            //For chunk boundary LCP calculation, to be done later.
            chunk_border_SA_info[i] = l1_sa_values_chunk(chunks[i], run_heads_bv, sa_chunks_tmp_files[i], lcp_chunks_tmp_files[i]);
            spdlog::info("Chunk {}/{} completed", i + 1, chunks.size());
        }
    
        // ----------
        // Merge sa values tmp files
        spdlog::info("Writing out sa values at run heads");
        std::ofstream out_sa_values(l1_prefix + ".ssa", std::ios::out | std::ios::binary);
        std::ofstream out_lcp_values(l1_prefix + ".slcp", std::ios::out | std::ios::binary);
        
        // write number of values first
        pfpds::long_type tot_sa_values = run_heads_bv.number_of_1();
        out_sa_values.write((char*) &tot_sa_values, sizeof(tot_sa_values));
        out_lcp_values.write((char*) &tot_sa_values, sizeof(tot_sa_values));
        
        for (pfpds::long_type i = 0; i < chunks.size(); i++)
        {
            std::ifstream if_sa_chunk(sa_chunks_tmp_files[i], std::ios_base::binary);
            std::ifstream if_lcp_chunk(lcp_chunks_tmp_files[i], std::ios_base::binary);
            out_sa_values << if_sa_chunk.rdbuf();
            out_lcp_values << if_lcp_chunk.rdbuf();
        }
        out_sa_values.close();
        out_lcp_values.close();
    
        spdlog::info("Done");
    }
    
};

}


#endif //rpfbwt_algorithm_hpp
