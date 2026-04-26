#pragma once
#include <cstdint>

namespace catan {

    struct BoardLayout { // Static after random initilasation
        // Hexes (19): board layout, randomized per episode. Adjacency is static in topology.hpp.
        uint8_t hex_resource[19];   // 0=brick 1=lumber 2=wool 3=grain 4=ore 5=desert
        uint8_t hex_number[19];     // 0 for desert; else 2..12 skipping 7
        uint8_t port_type[9];       // 0..4 = 2:1 specific resource, 5 = 3:1 generic; indexed by port slot

    };
    struct alignas(64) GameState {
        // Nodes (54): owner + level packed into one byte each.
        uint8_t node[54];

        // Edges (72): one byte per edge, owner id or EDGE_EMPTY.
        uint8_t edge[72];
        uint8_t robber_hex; // hex of current robber.
        
        // gamestate
        bool dice_rolled; // checks wether dice has be rolled, before dice only dev card can be played.
        // ["discard_resources", "move_robber", "robber_steal", "Year of Plenty", "Monopoly", "place_road", "Trade Pending", "accept_trade", "decline_trade", "confirm_trade", "end_trade"]:
        uint8_t FLAG; // Flag is set to a integer if a player or all players have a forced action.
        uint8_t init_placement[4]; // Sets the inital placement order and what player may start the game.


        // Per player (index 0..3).
        uint8_t player_resources[4][5];   // brick, lumber, wool, grain, ore
        uint8_t player_dev[4][5];             // development cards: knight, victory point, road building, year of plenty, monopoly
        uint8_t player_vp[4];             // victory points

        uint8_t player_knights_played[4];   // for largest army
        uint8_t player_road_length[4];  // for longest road

        uint8_t player_settlment_count[4]; // starts at 5, how many settlements left to place
        uint8_t player_city_count[4]; // starts at 4, how many cities left to place
        uint8_t player_road_count[4]; // starts at 15, how many roads left to place

        bool player_longest_road[4]; // has longest road
        bool player_largest_army[4]; // has largest army
        bool player_played_card[4]; // has played dev card this turn (max 1 per turn)
        bool current_turn[4]; // current players turn

        // public state other players can see
        uint8_t player_handsize[4]; // total hand
        uint8_t player_total_dev[4]; // total hidden def cards
        uint8_t player_vp_without_dev[4]; // total vps - player dev vps
    };
} 
