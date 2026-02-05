// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SafeCaller
 * @dev Contract demonstrating safe low-level call handling
 * @notice Always checks return values of external calls
 */
contract SafeCaller {
    address public owner;

    event CallSuccess(address target, bytes data, bytes returnData);
    event SendSuccess(address recipient, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    /// @notice Safe external call with return value check
    function safeCall(
        address target,
        bytes memory data
    ) public onlyOwner returns (bytes memory) {
        require(target != address(0), "Invalid target");

        // ALWAYS check return value
        (bool success, bytes memory returnData) = target.call(data);
        require(success, "Call failed");

        emit CallSuccess(target, data, returnData);
        return returnData;
    }

    /// @notice Safe send with return value check
    function safeSend(address payable recipient, uint256 amount) public onlyOwner {
        require(recipient != address(0), "Invalid recipient");
        require(amount <= address(this).balance, "Insufficient balance");

        // Use call instead of send for better gas handling
        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Send failed");

        emit SendSuccess(recipient, amount);
    }

    /// @notice Safe delegate call (very careful usage)
    function safeDelegateCall(
        address implementation,
        bytes memory data
    ) public onlyOwner returns (bytes memory) {
        require(implementation != address(0), "Invalid implementation");
        // Note: delegatecall should be used with extreme caution
        // Only call trusted, immutable implementations

        (bool success, bytes memory returnData) = implementation.delegatecall(data);
        require(success, "Delegatecall failed");

        return returnData;
    }

    receive() external payable {}
}
