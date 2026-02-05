// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title AccessControlled
 * @dev A contract with proper access control using msg.sender
 * @notice Demonstrates secure access control patterns (NOT using tx.origin)
 */
contract AccessControlled {
    address public owner;
    address public admin;
    mapping(address => bool) public authorized;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event AdminUpdated(address indexed previousAdmin, address indexed newAdmin);
    event FundsWithdrawn(address indexed to, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not the owner");
        _;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin || msg.sender == owner, "Caller is not admin");
        _;
    }

    modifier onlyAuthorized() {
        require(authorized[msg.sender] || msg.sender == owner, "Not authorized");
        _;
    }

    constructor() {
        owner = msg.sender;
        admin = msg.sender;
    }

    /// @notice Transfer ownership using msg.sender (secure)
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner is zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function setAdmin(address newAdmin) public onlyOwner {
        require(newAdmin != address(0), "New admin is zero address");
        emit AdminUpdated(admin, newAdmin);
        admin = newAdmin;
    }

    function authorize(address user) public onlyAdmin {
        authorized[user] = true;
    }

    function revoke(address user) public onlyAdmin {
        authorized[user] = false;
    }

    /// @notice Secure withdraw with proper access control
    function withdraw(address payable to, uint256 amount) public onlyOwner {
        require(to != address(0), "Cannot withdraw to zero address");
        require(amount <= address(this).balance, "Insufficient contract balance");

        (bool success, ) = to.call{value: amount}("");
        require(success, "Transfer failed");

        emit FundsWithdrawn(to, amount);
    }

    receive() external payable {}
}
