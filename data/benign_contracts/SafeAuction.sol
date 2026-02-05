// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SafeAuction
 * @dev Auction contract with pull-over-push payment pattern to prevent DoS
 * @notice Uses withdrawal pattern instead of automatic refunds
 */
contract SafeAuction {
    address public highestBidder;
    uint256 public highestBid;
    address public beneficiary;
    uint256 public auctionEndTime;
    bool public ended;

    // Pull-over-push: users must withdraw their own funds
    mapping(address => uint256) public pendingReturns;

    event HighestBidIncreased(address bidder, uint256 amount);
    event AuctionEnded(address winner, uint256 amount);
    event Withdrawal(address bidder, uint256 amount);

    constructor(address _beneficiary, uint256 _biddingTime) {
        beneficiary = _beneficiary;
        auctionEndTime = block.timestamp + _biddingTime;
    }

    function bid() public payable {
        require(block.timestamp < auctionEndTime, "Auction already ended");
        require(msg.value > highestBid, "Bid not high enough");

        if (highestBid > 0) {
            // Don't refund directly - add to pending returns (pull pattern)
            pendingReturns[highestBidder] += highestBid;
        }

        highestBidder = msg.sender;
        highestBid = msg.value;
        emit HighestBidIncreased(msg.sender, msg.value);
    }

    /// @notice Pull-over-push: users withdraw their own funds
    function withdraw() public {
        uint256 amount = pendingReturns[msg.sender];
        require(amount > 0, "Nothing to withdraw");

        // Set to zero BEFORE transfer (checks-effects-interactions)
        pendingReturns[msg.sender] = 0;

        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Transfer failed");

        emit Withdrawal(msg.sender, amount);
    }

    function endAuction() public {
        require(block.timestamp >= auctionEndTime, "Auction not yet ended");
        require(!ended, "Auction end already called");

        ended = true;
        emit AuctionEnded(highestBidder, highestBid);

        (bool success, ) = payable(beneficiary).call{value: highestBid}("");
        require(success, "Transfer to beneficiary failed");
    }
}
