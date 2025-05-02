#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <map>

/**
 * @brief Represents a directed edge in the tree with metadata.
 */
struct Edge{
    int from;    ///< Source node of the edge
    int to;      ///< Destination node
    int reverse; ///< Index of the reverse edge
    int next;    ///< Index of the next edge in Euler tour for reverse edge
};

/**
 * @brief Returns indices of all non-leaf nodes in a binary tree.
 * 
 * @param treeStr String representation of the binary tree.
 * @return Vector of non-leaf node indices.
 */
std::vector<int> getNonLeafIndices(const std::string &treeStr){
    std::vector<int> indices;
    for (int i = 0; i < (int)treeStr.size(); ++i){
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left < (int)treeStr.size() || right < (int)treeStr.size()){
            indices.push_back(i);
        }
    }
    return indices;
}

/**
 * @brief Assigns one non-leaf node index to each MPI process.
 * 
 * @param rank Process rank.
 * @param k Total number of non-leaf nodes.
 * @param nonLeafIndices Vector of non-leaf node indices.
 * @return Index of the node assigned to this process.
 */
int assignNodeToProcess(int rank, int k, const std::vector<int> &nonLeafIndices){
    int assigned = -1;
    if (rank == 0){
        for (int i = 1; i < k; ++i){
            MPI_Send(&nonLeafIndices[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        assigned = nonLeafIndices[0];
    }
    else if (rank < k){
        MPI_Recv(&assigned, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return assigned;
}

/**
 * @brief Constructs local edges and adjacency list for the assigned node.
 * 
 * @param assignedNode Index of the node assigned to this process.
 * @param rank Process rank.
 * @param n Total number of nodes.
 * @param flatEdges Flattened representation of the edges.
 * @param localAdjList Local adjacency list.
 */
void buildLocalEdgesAndAdjList(int assignedNode, int rank, int n, std::vector<int> &flatEdges, std::map<int, std::vector<int>> &localAdjList){
    std::vector<Edge> edges;
    int left = 2 * assignedNode + 1;
    int right = 2 * assignedNode + 2;

    if (left < n){
        edges.push_back({assignedNode, left, -1, -1});
        edges.push_back({left, assignedNode, -1, -1});
    }
    if (right < n){
        edges.push_back({assignedNode, right, -1, -1});
        edges.push_back({right, assignedNode, -1, -1});
    }

    for (size_t i = 0; i < edges.size(); ++i){
        int globalIdx = rank * 4 + i;
        edges[i].reverse = (i % 2 == 0) ? globalIdx + 1 : globalIdx - 1;
        flatEdges.push_back(edges[i].from);
        flatEdges.push_back(edges[i].to);
        flatEdges.push_back(edges[i].reverse);
        flatEdges.push_back(edges[i].next);

        if (edges[i].from == assignedNode){
            localAdjList[assignedNode].push_back(globalIdx);
        }
        else{
            localAdjList[edges[i].from].push_back(globalIdx);
        }
    }
}

/**
 * @brief Flattens the local adjacency list into a vector.
 * 
 * @param localAdj Local adjacency list.
 * @return Flattened vector.
 */
std::vector<int> flattenAdjList(const std::map<int, std::vector<int>> &localAdj){
    std::vector<int> flat;
    for (const auto &[from, vec] : localAdj){
        flat.push_back(from);
        flat.push_back(vec.size());
        for (int i : vec)
            flat.push_back(i);
    }
    return flat;
}

/**
 * @brief Gathers graph data (edges and adjacency lists) from all processes to process 0.
 * 
 * @param flatEdges Local flattened edges.
 * @param flatAdj Local flattened adjacency list.
 * @param size Total number of processes.
 * @param allEdges Combined edge data on process 0.
 * @param allAdj Combined adjacency list data on process 0.
 * @param edgeCount Output edge count.
 * @param recvCounts Vector to hold the receive counts.
 */
void gatherGraphData(const std::vector<int> &flatEdges, const std::vector<int> &flatAdj, int size,std::vector<int> &allEdges, std::vector<int> &allAdj, int &edgeCount, std::vector<int> &recvCounts){
    int localCount = flatEdges.size();
    recvCounts.resize(size);
    MPI_Gather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(size);
    int total = 0;
    for (int i = 0; i < size; ++i){
        displs[i] = total;
        total += recvCounts[i];
    }
    allEdges.resize(total);
    MPI_Gatherv(flatEdges.data(), localCount, MPI_INT, allEdges.data(), recvCounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    int localAdjSize = flatAdj.size();
    std::vector<int> adjSizes(size);
    MPI_Gather(&localAdjSize, 1, MPI_INT, adjSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> adjDispls(size);
    int totalAdj = 0;
    for (int i = 0; i < size; ++i){
        adjDispls[i] = totalAdj;
        totalAdj += adjSizes[i];
    }
    allAdj.resize(totalAdj);
    MPI_Gatherv(flatAdj.data(), localAdjSize, MPI_INT, allAdj.data(), adjSizes.data(), adjDispls.data(), MPI_INT, 0, MPI_COMM_WORLD);

    edgeCount = total / 4;
}

/**
 * @brief Reconstructs the graph from gathered data.
 * 
 * @param allEdges Flattened edge data.
 * @param allAdj Flattened adjacency list data.
 * @param edges Output vector of Edge structs.
 * @param adjList Output adjacency list.
 */
void reconstructGraph(const std::vector<int> &allEdges, const std::vector<int> &allAdj,std::vector<Edge> &edges, std::map<int, std::vector<int>> &adjList){
    for (size_t i = 0; i + 3 < allEdges.size(); i += 4){
        edges.push_back({allEdges[i], allEdges[i + 1], allEdges[i + 2], allEdges[i + 3]});
    }
    for (size_t i = 0; i < allAdj.size();){
        int from = allAdj[i++];
        int count = allAdj[i++];
        while (count--){
            adjList[from].push_back(allAdj[i++]);
        }
    }
}

/**
 * @brief Assigns 'next' pointers for Euler tour traversal.
 * 
 * @param edges Vector of edges.
 * @param adjList Adjacency list.
 */
void assignNextPointers(std::vector<Edge>& edges, const std::map<int, std::vector<int>>& adjList) {
    for (auto& e : edges) {
        int rev = e.reverse;
        if (rev != -1) {
            const auto& adj = adjList.at(edges[rev].from);
            for (size_t i = 0; i + 1 < adj.size(); ++i) {
                if (adj[i] == rev) {
                    e.next = adj[i + 1];
                    break;
                }
            }
        }
    }
}

/**
 * @brief Sends Euler tour tasks (edge + adjacency) to worker processes.
 * 
 * @param edges Vector of edges.
 * @param adjList Adjacency list.
 * @param maxEdges Number of edges to distribute.
 * @param size Total number of processes.
 */
void sendEtourTasks(const std::vector<Edge>& edges, const std::map<int, std::vector<int>>& adjList, int maxEdges, int size) {
    for (int i = 0; i < maxEdges; ++i) {
        const Edge& e = edges[i];
        std::vector<int> adj = adjList.at(e.to);
        int info[5] = {e.from, e.to, e.reverse, e.next, (int)adj.size()};
        MPI_Send(info, 5, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
        for (int idx : adj)
            MPI_Send(&idx, 1, MPI_INT, i + 1, 2, MPI_COMM_WORLD);
    }
}

/**
 * @brief Worker process receives Euler tour task and sends computed etour index back.
 * 
 * @param rank Process rank.
 * @param edgeCount Total number of edges.
 */
void receiveEtour(int rank, int edgeCount) {
    int info[5];
    MPI_Recv(info, 5, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int next = info[3], adjSize = info[4];
    std::vector<int> adj(adjSize);
    for (int i = 0; i < adjSize; ++i) {
        MPI_Recv(&adj[i], 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    int etour = (next != -1) ? next : (adjSize > 0 ? adj[0] : -1);
    MPI_Send(&etour, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
}
/**
 * @brief Gathers Euler tour results and generates edge weights for WLA.
 * 
 * @param edgeCount Number of edges.
 * @param size Total number of processes.
 * @param etour Output vector of etour next indices.
 * @param weights Output vector of edge weights.
 * @param edges Edge list.
 */
void gatherEtourAndWeights(int edgeCount, int size, std::vector<int>& etour, std::vector<int>& weights, const std::vector<Edge>& edges) {
    for (int i = 0; i < edgeCount; ++i) {
        if (i + 1 < size) {
            int val;
            MPI_Recv(&val, 1, MPI_INT, i + 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            etour[i] = val;
        }
    }
    for (int i = 0; i < edgeCount; ++i) {
        weights[i] = (i % 2 == 0) ? -1 : 1;
    }
}
/**
 * @brief Sends edge data and broadcasts etour and weights to workers.
 * 
 * @param edges Edge list.
 * @param etour Euler tour indices.
 * @param weights Edge weights.
 * @param edgeCount Total number of edges.
 * @param maxEdges Number of edges handled in parallel.
 * @param size Total number of processes.
 */
void sendForwardEdges(const std::vector<Edge>& edges, const std::vector<int>& etour, const std::vector<int>& weights, int edgeCount, int maxEdges, int size) {
    for (int i = 0; i < maxEdges; ++i) {
        if (i % 2 == 0 && i + 1 < size) {
            const Edge& e = edges[i];
            int info[5] = {e.from, e.to, e.reverse, etour[i], weights[i]};
            MPI_Send(info, 5, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Bcast((void*)etour.data(), edgeCount, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)weights.data(), edgeCount, MPI_INT, 0, MPI_COMM_WORLD);
}

/**
 * @brief Computes node levels in the tree using WLA algorithm on each worker.
 * 
 * @param rank Process rank.
 * @param size Total number of processes.
 * @param edgeCount Total number of edges.
 */
void computeLevelsOnWorkers(int rank, int size, int edgeCount) {
    std::vector<int> fullEtour(edgeCount);
    std::vector<int> fullWeights(edgeCount);
    if (edgeCount > 0) {
        MPI_Bcast(fullEtour.data(), edgeCount, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(fullWeights.data(), edgeCount, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank > 0 && rank < size && (rank - 1) % 2 == 0) {
        int info[5];
        MPI_Recv(info, 5, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int etour = info[3], weight = info[4];

        int sum = weight;
        int next = etour;
        while (next != 0 && next != -1) {
            sum += fullWeights[next];
            next = fullEtour[next];
        }

        int level = sum + 1;
        MPI_Send(&level, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
}

/**
 * @brief Receives levels from worker processes and prints final level assignments.
 * 
 * @param rank Process rank.
 * @param treeStr String representation of the tree.
 * @param edges Edge list.
 * @param maxEdges Number of forward edges.
 */
void printFinalLevels(int rank, const std::string& treeStr, const std::vector<Edge>& edges, int maxEdges) {
    std::vector<std::string> parts;
    parts.push_back(std::string(1, treeStr[0]) + ":0");

    for (size_t i = 0; i < edges.size(); ++i) {
        if (i % 2 == 0 && i < maxEdges) {
            int lvl;
            MPI_Recv(&lvl, 1, MPI_INT, i + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            parts.push_back(std::string(1, treeStr[edges[i].to]) + ":" + std::to_string(lvl));
        }
    }

    for (size_t i = 0; i < parts.size(); ++i) {
        std::cout << parts[i];
        if (i + 1 < parts.size())
            std::cout << ",";
    }
    std::cout << std::endl;
}

/**
 * @brief Main entry point. Orchestrates the parallel computation of node levels in a binary tree.
 * 
 * @param argc Argument count.
 * @param argv Argument vector. Expects one argument: the tree string.
 * @return int Exit status.
 */
int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2){
        if (rank == 0){
            std::cerr << "Usage: " << argv[0] << " <tree_string>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string treeStr = argv[1];
    int n = treeStr.size();
    std::vector<int> nonLeafIndices = getNonLeafIndices(treeStr);

    int k = nonLeafIndices.size();
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int assignedNode = assignNodeToProcess(rank, k, nonLeafIndices);

    std::vector<int> flatEdges, flatAdj;
    std::map<int, std::vector<int>> localAdjList;
    if (rank < k){
        buildLocalEdgesAndAdjList(assignedNode, rank, n, flatEdges, localAdjList);
    }
    flatAdj = flattenAdjList(localAdjList);

    std::vector<int> allEdges, allAdj, recvCounts;
    int edgeCount = 0;
    gatherGraphData(flatEdges, flatAdj, size, allEdges, allAdj, edgeCount, recvCounts);

    std::vector<Edge> edges;
    std::map<int, std::vector<int>> adjList;
    if (rank == 0){
        reconstructGraph(allEdges, allAdj, edges, adjList);
        assignNextPointers(edges, adjList);
    }

    edgeCount = edges.size();
    int maxEdges = std::min(edgeCount, size - 1);
    MPI_Bcast(&edgeCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0){
        sendEtourTasks(edges, adjList, maxEdges, size);
    }

    if (rank > 0 && rank < edgeCount + 1){
        receiveEtour(rank, edgeCount);
    }

    std::vector<int> etour(edgeCount);
    std::vector<int> weights(edgeCount, 0);

    if (rank == 0){
        gatherEtourAndWeights(edgeCount, size, etour, weights, edges);
        sendForwardEdges(edges, etour, weights, edgeCount, maxEdges, size);
    }

    computeLevelsOnWorkers(rank, size, edgeCount);

    if (rank == 0){
        printFinalLevels(rank, treeStr, edges, maxEdges);
    }
    MPI_Finalize();
    return 0;
}