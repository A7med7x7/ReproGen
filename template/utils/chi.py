from chi import lease

def get_active_lease(lease_name=None, project_name=None):
    """
    This function Return an active lease by exact name or by project name prefix.
    if lease_name is given it returns the lease otherwise it lists ACTIVE leases 
    starting with project_name. it raises an error if none or multiple matches are found.
    """
    leases = lease.list_leases()

    # retrieve exact lease name provided
    if lease_name:
        match = next((l for l in leases if l.name == lease_name), None)
        if not match:
            raise ValueError(f"No lease found with name '{lease_name}'")
        return match

    matching = [
        l for l in leases
        if l.name.startswith(project_name) and l.status.upper() == "ACTIVE"
    ]

    if not matching:
        raise ValueError(f"No active lease found starting with '{project_name}'")

    if len(matching) > 1:
        print("\nYou have multiple active leases:")
        for l in matching:
            print(f" - {l.name} (status: {l.status})")
        raise ValueError("Please set 'lease_name' to pick the correct lease.")

    return matching[0]
