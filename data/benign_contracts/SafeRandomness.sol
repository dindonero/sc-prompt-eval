// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SafeRandomness
 * @dev Contract that uses commit-reveal for fair randomness
 * @notice Demonstrates secure randomness pattern (no block.timestamp abuse)
 */
contract SafeRandomness {
    struct Commitment {
        bytes32 hash;
        uint256 revealDeadline;
        bool revealed;
        uint256 value;
    }

    mapping(address => Commitment) public commitments;
    mapping(address => uint256) public rewards;

    uint256 public constant REVEAL_PERIOD = 1 hours;

    event Committed(address indexed user, bytes32 hash);
    event Revealed(address indexed user, uint256 value);
    event RewardClaimed(address indexed user, uint256 amount);

    /// @notice Commit a hashed value (user commits before revealing)
    function commit(bytes32 hash) public payable {
        require(msg.value >= 0.01 ether, "Minimum stake required");
        require(commitments[msg.sender].hash == bytes32(0), "Already committed");

        commitments[msg.sender] = Commitment({
            hash: hash,
            revealDeadline: block.timestamp + REVEAL_PERIOD,
            revealed: false,
            value: 0
        });

        emit Committed(msg.sender, hash);
    }

    /// @notice Reveal the committed value
    function reveal(uint256 value, bytes32 salt) public {
        Commitment storage c = commitments[msg.sender];
        require(c.hash != bytes32(0), "No commitment found");
        require(!c.revealed, "Already revealed");
        require(block.timestamp <= c.revealDeadline, "Reveal period expired");
        require(keccak256(abi.encodePacked(value, salt)) == c.hash, "Invalid reveal");

        c.revealed = true;
        c.value = value;

        // Reward based on revealed value (just an example)
        if (value % 7 == 0) {
            rewards[msg.sender] = 0.02 ether;
        }

        emit Revealed(msg.sender, value);
    }

    function claimReward() public {
        uint256 reward = rewards[msg.sender];
        require(reward > 0, "No reward to claim");

        rewards[msg.sender] = 0;

        (bool success, ) = payable(msg.sender).call{value: reward}("");
        require(success, "Transfer failed");

        emit RewardClaimed(msg.sender, reward);
    }

    function getCommitmentHash(uint256 value, bytes32 salt) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(value, salt));
    }
}
