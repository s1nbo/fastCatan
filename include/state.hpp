#pragma once
#include <cstdint>
#include <type_traits>

namespace catan {

    // player-id fields when no player owns the slot
    inline constexpr uint8_t NO_PLAYER = 0xFF;

    // ---------------------------------------------------------------------
    // Phase: top-level game stage. Drives which actions are legal.
    // ---------------------------------------------------------------------
    enum class Phase : uint8_t {
        INITIAL_PLACEMENT_1 = 0, // forward order: each player places 1st settlement + road
        INITIAL_PLACEMENT_2,     // reverse order: each player places 2nd settlement + road
        MAIN,                    // normal turns: roll dice, build, trade, dev cards
        ENDED,                   // terminal: a player reached 10 VP
    };

    // ---------------------------------------------------------------------
    // Flag: forced-action override. NONE during normal play; set when a
    // player (or all players) must resolve a specific action before the
    // game can continue.
    // ---------------------------------------------------------------------
    enum class Flag : uint8_t {
        NONE = 0,
        DISCARD_RESOURCES, // players with >7 cards must discard half (after a 7 roll)
        MOVE_ROBBER,       // current player must move robber
        ROBBER_STEAL,      // current player must pick victim to steal from
        YEAR_OF_PLENTY,    // current player picks 2 resources from bank
        MONOPOLY,          // current player picks a resource type
        PLACE_ROAD,        // road building dev card: place up to 2 free roads
        TRADE_PENDING,     // current player proposed trade, awaiting responses
    };

    // ---------------------------------------------------------------------
    // BoardLayout: static after random initialisation at episode start.
    // Topology / adjacency lives in topology.hpp.
    // ---------------------------------------------------------------------
    struct BoardLayout {
        uint8_t hex_resource[19]; // 0=brick 1=lumber 2=wool 3=grain 4=ore 5=desert
        uint8_t hex_number[19];   // 0 for desert; else 2..12 skipping 7
        uint8_t port_type[9];     // 0..4 = 2:1 specific resource, 5 = 3:1 generic
    };

    // ---------------------------------------------------------------------
    // GameState: dynamic per-step state. 64-byte aligned, 256 bytes total
    // ---------------------------------------------------------------------
    struct alignas(64) GameState {
        // --- Board state ---
        uint8_t node[54];   // owner + building level packed into one byte each
        uint8_t edge[72];   // owner id or EDGE_EMPTY
        uint8_t robber_hex; // hex index of current robber

        // --- Turn / phase state ---
        uint8_t  dice_roll;    // 0 if not yet rolled this turn, else 2..12
        uint16_t turn_count;   // monotonic turn counter (termination cap)
        Phase    phase;        // INITIAL_PLACEMENT_1 / _2 / MAIN / ENDED
        Flag     flag;         // forced-action override; NONE during normal play
        uint8_t  start_player; // who began initial placement; phase 1 clockwise from here, phase 2 counter-clockwise back to here

        // --- Per-player private state (index 0..3) ---
        uint8_t player_resources[4][5];            // brick, lumber, wool, grain, ore
        uint8_t player_dev[4][5];                  // dev cards: knight, VP, road building, year of plenty, monopoly
        uint8_t player_dev_bought_this_turn[4][5]; // one-turn cooldown on newly bought dev cards (VP exempt)
        uint8_t player_vp[4];                      // total victory points
        uint8_t player_ports[4][6];                // brick, lumber, wool, grain, ore, 3:1

        // --- Per-player counters ---
        uint8_t player_knights_played[4];   // contributes to largest army
        uint8_t player_road_length[4];      // longest contiguous road
        uint8_t player_settlement_count[4]; // settlements left to place (starts at 5)
        uint8_t player_city_count[4];       // cities left to place      (starts at 4)
        uint8_t player_road_count[4];       // roads left to place       (starts at 15)

        // --- Awards / turn flags ---
        uint8_t longest_road_owner; // player id, or sentinel if none
        uint8_t largest_army_owner; // player id, or sentinel if none
        bool    dev_card_played;    // current player already played a dev card this turn
        uint8_t current_player;     // whose turn it is

        // --- Per-player public state (visible to all players) ---
        uint8_t player_handsize[4];       // total resource cards held
        uint8_t player_total_dev[4];      // total hidden dev cards held
        uint8_t player_vp_without_dev[4]; // public VP = total VP minus hidden dev-card VPs

        // --- Bank ---
        uint8_t bank[5];     // resource supply: brick, lumber, wool, grain, ore
        uint8_t dev_deck[5]; // dev cards remaining: knight, VP, road building, year of plenty, monopoly
    };

    // Just for Checking.
    static_assert(std::is_trivially_copyable_v<GameState>,
                  "GameState must be trivially copyable for memcpy cloning");
    static_assert(alignof(GameState) == 64,
                  "GameState must be 64-byte aligned (cache line)");
    // static_assert(sizeof(GameState) == 256,
                  // "GameState must be exactly 4 cache lines; update if layout changes intentionally");
    static_assert(std::is_trivially_copyable_v<BoardLayout>,
                  "BoardLayout must be trivially copyable");

}

/*
GAPS:
1. Discard tracking (7-roll) — All players with >7 cards must discard, not just current. Need per-player pending count:                                                                     
  uint8_t player_discard_owed[4];  // cards still to discard, 0 = done                                                                                                                        
  Without this, can't resume after multi-player discard.                                                                                                                                      
                                                                                                                                                                                              
  2. Road Building dev card — Up to 2 free roads, sequential placement. Need:                                                                                                                 
  uint8_t free_roads_remaining;  // 0..2 when Flag::PLACE_ROAD active                                                                                                                         
                                                                                                                                                                                              
  3. Year of Plenty progress — If modeled as 2 sub-picks. If single atomic action (pick both at once) → no field needed. Decide action model.                                                 
                                                                                                                                                                                              
  4. Trade offer state — Flag::TRADE_PENDING exists but no fields. If supporting open trades:                                                                                                 
  uint8_t trade_give[5], trade_want[5];                                                                                                                                                       
  uint8_t trade_proposer;                                                                                                                                                                     
  uint8_t trade_responses;  // bitmask: bit i = player i responded
  Skip if action model = instant 1-on-1 swap.                     
                                                                                                                                                                                              
  5. num_players — Hardcoded 4. If supporting 3-player games, need uint8_t num_players;. Otherwise fine, document "always 4."                                                                 
                                                                                                                                                                                              
  6. Initial placement direction tracking — start_player + current_player + phase. Phase 1 forward, phase 2 reverse. Order derivable. OK if step logic implements direction switch correctly. 
  No field gap.                                                                                                                                                                               
                                                                                                                                                                                              
  Smaller nits                                                                                                                                                                                
                                       
  - dev_card_played bool — fine. Compiler treats as 1 byte.                                                                                                                                   
  - node[54] packing convention not documented — what bits = owner, what = level (settlement/city)? Add comment showing layout (e.g., bits 0-1 = owner, bit 2 = is_city, or similar).
  - edge[72] empty sentinel — uses EDGE_EMPTY in comment but constant not defined. Only NO_PLAYER = 0xFF exists. Either rename to one constant or define EDGE_EMPTY.                          
  - dev_deck[5] order — confirm matches player_dev[4][5] order (knight, VP, RB, YoP, monopoly). Comment matches; sanity check at use site.                                                    
  - player_settlement_count — comment says "settlements left to place" (decrements from 5). Earlier version was "for tie-breaking in longest road" (count placed). Pick one meaning, stay     
  consistent. Tie-break needs placed count. Currently using remaining. If using remaining: tie-break = 5 - remaining. Document.                                                               
  - Same applies to player_city_count, player_road_count.                                                                                                                                     
                                                                                                                                                                                              
  Sanity                                                                                                                                                                                      
                                                                                                                                                                                              
  sizeof math:                                                                                                                                                                                
  - Members sum ≈ 243 bytes
  - alignas(64) rounds to 256. static_assert holds.                                                                                                                                           
                                       
  Verdict                                                                                                                                                                                     
                                       
  Functionally close to ready. Add discard_owed + free_roads_remaining and you cover all forced-action mid-state resumes. Trade fields depend on action model. Define EDGE_EMPTY, document    
  node packing, lock down placed-vs-remaining semantics. Good to wire up step logic and let profiler guide rest.


*/