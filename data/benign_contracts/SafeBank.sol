// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SafeBank
 * @dev A secure bank contract using Checks-Effects-Interactions pattern
 * @notice This contract demonstrates reentrancy-safe withdrawal
 */
contract SafeBank {
    mapping(address => uint256) public balances;
    address public owner;

    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function deposit() public payable {
        require(msg.value > 0, "Must deposit positive amount");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    /// @notice Secure withdrawal using Checks-Effects-Interactions
    function withdraw(uint256 amount) public {
        // CHECKS
        require(balances[msg.sender] >= amount, "Insufficient balance");

        // EFFECTS (state update BEFORE external call)
        balances[msg.sender] -= amount;

        // INTERACTIONS (external call AFTER state update)
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Transfer failed");

        emit Withdrawal(msg.sender, amount);
    }

    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}
