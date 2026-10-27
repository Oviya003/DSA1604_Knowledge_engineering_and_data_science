problem = input("Enter problem (slow/no internet/dns): ").lower()

if problem == "slow":
    print("Cause: Network congestion")
    print("Solution: Restart router and reduce network usage")
elif problem == "no internet":
    print("Cause: Router or connection failure")
    print("Solution: Check cables and restart router")
elif problem == "dns":
    print("Cause: DNS resolution failure")
    print("Solution: Change DNS server")
else:
    print("Problem not identified")
