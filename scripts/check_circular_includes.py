import os
import re


def find_metal_files(root_dir):
    metal_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".metal"):
                metal_files.append(os.path.normpath(os.path.join(root, file)))
    return metal_files

def get_includes(file_path, root_dir):
    includes = []
    try:
        with open(file_path) as f:
            for line in f:
                # Flexible regex for #include "..."
                match = re.search(r'^\s*#include\s+"([^"]+)"', line)
                if match:
                    inc_name = match.group(1)
                    # Resolve include path relative to the file
                    dir_path = os.path.dirname(file_path)
                    inc_path = os.path.normpath(os.path.join(dir_path, inc_name))

                    if os.path.exists(inc_path):
                        includes.append(inc_path)
                    else:
                        # Try relative to src/ if not found relative to file
                        src_path = os.path.normpath(os.path.join("contrib/metal_marlin/src", inc_name))
                        if os.path.exists(src_path):
                            includes.append(src_path)
                        else:
                            # Try relative to the root_dir
                            root_inc_path = os.path.normpath(os.path.join(root_dir, inc_name))
                            if os.path.exists(root_inc_path):
                                includes.append(root_inc_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return includes

def find_all_cycles(graph):
    cycles = []
    visited = set()
    path = []
    path_set = set()

    def visit(u):
        if u in path_set:
            cycle_start_index = path.index(u)
            cycles.append(path[cycle_start_index:] + [u])
            return

        if u in visited:
            return

        visited.add(u)
        path.append(u)
        path_set.add(u)

        for v in graph.get(u, []):
            visit(v)

        path.pop()
        path_set.remove(u)

    for node in list(graph.keys()):
        visit(node)
    return cycles

def main():
    root_dir = "contrib/metal_marlin"
    metal_files = find_metal_files(root_dir)

    graph = {}
    for f in metal_files:
        graph[f] = get_includes(f, root_dir)

    cycles = find_all_cycles(graph)
    if cycles:
        print(f"FOUND {len(cycles)} CIRCULAR INCLUDE(S):")
        for cycle in cycles:
            print(" -> ".join(cycle))
        exit(1)
    else:
        # Debug: print some detected includes to be sure
        print("No circular includes found.")
        print("\nSample includes found:")
        count = 0
        for f, incs in graph.items():
            if incs:
                print(f"{f} includes: {', '.join(incs)}")
                count += 1
                if count > 5:
                    break
        exit(0)

if __name__ == "__main__":
    main()
