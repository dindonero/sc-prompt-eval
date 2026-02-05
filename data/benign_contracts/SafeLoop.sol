// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SafeLoop
 * @dev Contract with bounded loops to prevent DoS
 * @notice Uses pagination pattern for gas-safe iteration
 */
contract SafeLoop {
    address public owner;
    address[] private users;
    mapping(address => uint256) public balances;

    // Maximum users per batch to prevent gas limit issues
    uint256 public constant MAX_BATCH_SIZE = 100;

    event UserAdded(address user);
    event DistributionCompleted(uint256 from, uint256 to, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function addUser(address user) public onlyOwner {
        require(user != address(0), "Invalid user");
        users.push(user);
        emit UserAdded(user);
    }

    function getUserCount() public view returns (uint256) {
        return users.length;
    }

    /// @notice Distribute funds in batches to prevent DoS
    /// @param startIndex Starting index for this batch
    /// @param amount Amount per user
    function distributeBatch(
        uint256 startIndex,
        uint256 amount
    ) public onlyOwner {
        require(startIndex < users.length, "Invalid start index");

        // Calculate end index with bounds checking
        uint256 endIndex = startIndex + MAX_BATCH_SIZE;
        if (endIndex > users.length) {
            endIndex = users.length;
        }

        // Bounded loop - maximum MAX_BATCH_SIZE iterations
        for (uint256 i = startIndex; i < endIndex; i++) {
            balances[users[i]] += amount;
        }

        emit DistributionCompleted(startIndex, endIndex, amount);
    }

    /// @notice Get users in a range (paginated)
    function getUsers(
        uint256 start,
        uint256 count
    ) public view returns (address[] memory) {
        require(start < users.length, "Start out of bounds");

        uint256 end = start + count;
        if (end > users.length) {
            end = users.length;
        }

        address[] memory result = new address[](end - start);
        for (uint256 i = start; i < end; i++) {
            result[i - start] = users[i];
        }

        return result;
    }

    receive() external payable {}
}
