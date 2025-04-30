#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <map>

struct Edge {
    int from;
    int to;
    int reverse; 
    int next;    
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <tree_string>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string treeStr = argv[1];
    int n = treeStr.size();

    std::vector<int> nonLeafIndices;
    if (rank == 0) {
        std::cout << "Total nodes: " << n << "\n";

        for (int i = 0; i < n; ++i) {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            if (left < n || right < n) {
                nonLeafIndices.push_back(i);
            }
        }

        std::cout << "Non-leaf node indices: ";
        for (int idx : nonLeafIndices) {
            std::cout << idx << " (" << treeStr[idx] << ") ";
        }
        std::cout << "\n";
    }

    int k = 0;
    if (rank == 0) k = nonLeafIndices.size();
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int assignedNode = -1;
    if (rank < k) {
        if (rank == 0) {
            for (int i = 1; i < k; ++i) {
                MPI_Send(&nonLeafIndices[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
            assignedNode = nonLeafIndices[0];
        } else {
            MPI_Recv(&assignedNode, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    std::vector<int> flatEdges;
    std::map<int, std::vector<int>> localAdjList;
    if (rank < k) {
        std::vector<Edge> edges;

        int left = 2 * assignedNode + 1;
        int right = 2 * assignedNode + 2;

        if (left < n) {
            edges.push_back({assignedNode, left, -1, -1});
            edges.push_back({left, assignedNode, -1, -1});
        }
        if (right < n) {
            edges.push_back({assignedNode, right, -1, -1});
            edges.push_back({right, assignedNode, -1, -1});
        }

        std::cout << "Rank " << rank << " assigned to node " << treeStr[assignedNode] << " (" << assignedNode << ")\n";
        for (size_t i = 0; i < edges.size(); ++i) {
            int globalIndex = rank * 4 + i;
            edges[i].reverse = (i % 2 == 0) ? (globalIndex + 1) : (globalIndex - 1);
            std::cout << "  Edge: " << treeStr[edges[i].from] << " -> " << treeStr[edges[i].to] << " (rev: " << edges[i].reverse << ")\n";
            flatEdges.push_back(edges[i].from);
            flatEdges.push_back(edges[i].to);
            flatEdges.push_back(edges[i].reverse);
            flatEdges.push_back(edges[i].next);

            // Добавляем в локальный AdjList
            if (edges[i].from == assignedNode) {
                localAdjList[assignedNode].push_back(globalIndex);
            } else {
                localAdjList[edges[i].from].push_back(globalIndex);
            }

            std::cout << "Local AdjList on rank " << rank << ":\n";
            for (const auto& [from, list] : localAdjList) {
                std::cout << "  " << treeStr[from] << " (" << from << ") -> ";
                for (int idx : list) std::cout << idx << " ";
                std::cout << "\n";
            }
        }
    }

    // Отправка localAdjList на rank 0
    std::vector<int> flatAdj;
    for (const auto& [from, vec] : localAdjList) {
        flatAdj.push_back(from);
        flatAdj.push_back(vec.size());
        for (int idx : vec) flatAdj.push_back(idx);
    }

    int localAdjSize = flatAdj.size();
    std::vector<int> adjSizes(size);
    MPI_Gather(&localAdjSize, 1, MPI_INT, adjSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> adjDispls(size);
    int totalAdj = 0;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            adjDispls[i] = totalAdj;
            totalAdj += adjSizes[i];
        }
    }

    std::vector<int> allAdj(totalAdj);
    MPI_Gatherv(flatAdj.data(), localAdjSize, MPI_INT, allAdj.data(), adjSizes.data(), adjDispls.data(), MPI_INT, 0, MPI_COMM_WORLD);

    int localCount = flatEdges.size();
    std::vector<int> recvCounts(size);
    MPI_Gather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(size);
    int total = 0;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            displs[i] = total;
            total += recvCounts[i];
        }
    }

    std::vector<int> allEdges(total);
    MPI_Gatherv(flatEdges.data(), localCount, MPI_INT, allEdges.data(), recvCounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<Edge> edges;
    std::map<int, std::vector<int>> adjList;

    if (rank == 0) {
        for (size_t i = 0; i + 3 < allEdges.size(); i += 4) {
            edges.push_back({allEdges[i], allEdges[i+1], allEdges[i+2], -1});
        }

        for (size_t i = 0; i < allAdj.size();) {
            int from = allAdj[i++];
            int count = allAdj[i++];
            while (count--) {
                adjList[from].push_back(allAdj[i++]);
            }
        }

        std::cout << "\nAdjacency list:\n";
        for (const auto& [from, list] : adjList) {
            std::cout << "  " << treeStr[from] << " (" << from << ") -> ";
            for (int idx : list) {
                std::cout << idx << " ";
            }
            std::cout << "\n";
        }

        // Правильное назначение next на основе reverse и AdjList
        for (size_t i = 0; i < edges.size(); ++i) {
            int revIdx = edges[i].reverse;
            if (revIdx != -1) {
                int fromOfRev = edges[revIdx].from;
                const auto& adj = adjList[fromOfRev];
                for (size_t j = 0; j + 1 < adj.size(); ++j) {
                    if (adj[j] == revIdx) {
                        edges[i].next = adj[j + 1];
                        break;
                    }
                }
            }
        }

        std::cout << "\nCollected edges on rank 0:\n";
        for (size_t i = 0; i < edges.size(); ++i) {
            std::cout << "  " << treeStr[edges[i].from] << " -> " << treeStr[edges[i].to]
                      << "  (rev: " << edges[i].reverse << ", next: " << edges[i].next << ")\n";
        }
    }

    MPI_Finalize();
    return 0;
}
