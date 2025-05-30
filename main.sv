// Pipeline_CPU_Core Module

`timescale 1ns / 1ps



// ALU Module
module ALU (
    input [31:0] operand1,
    input [31:0] operand2,
    input [3:0] alu_control,
    output reg [31:0] alu_result,
    output zero_flag
);
    always @(*) begin
        case (alu_control)
            4'h0: alu_result = operand1 + operand2; // ADD
            4'h1: alu_result = operand1 - operand2; // SUB
            4'h2: alu_result = operand1 & operand2; // AND
            4'h3: alu_result = operand1 | operand2; // OR
            4'h4: alu_result = operand1 ^ operand2; // XOR
            4'h5: alu_result = operand1 << operand2[4:0]; // SLL
            4'h6: alu_result = operand1 >> operand2[4:0]; // SRL
            4'h7: alu_result = $signed(operand1) >>> operand2[4:0]; // SRA
            4'h8: alu_result = ($signed(operand1) < $signed(operand2)) ? 32'h1 : 32'h0; // SLT
            4'h9: alu_result = (operand1 < operand2) ? 32'h1 : 32'h0; // SLTU
            default: alu_result = 32'h0; // Default to 0 to prevent 'x'
        endcase
    end

    assign zero_flag = (alu_result == 32'h0) ? 1'b1 : 1'b0;
endmodule

// Instruction_Decoder Module - RV32I Only (with csr_op)
module Instruction_Decoder (
    input [31:0] instruction,
    output reg [3:0] alu_control,
    output reg [4:0] rd,
    output reg [4:0] rs1,
    output reg [4:0] rs2,
    output reg [31:0] immediate,
    output reg reg_write_enable,
    output reg alu_src,
    output reg branch,
    output reg mem_read,
    output reg mem_write,
    output reg mem_to_reg,
    // Removed extension-related control signals
    output reg is_fence,
    output reg is_ecall,
    output reg is_ebreak,
    output reg is_csr,
    output reg [2:0] csr_op  // Added csr_op port
    // Removed is_mul and is_div as they pertain to extensions
);
    wire [6:0] opcode;
    wire [2:0] funct3;
    wire [6:0] funct7;

    assign opcode = instruction[6:0];
    assign funct3 = instruction[14:12];
    assign funct7 = instruction[31:25];

    // Opcode constants for RV32I
    parameter OPCODE_R_TYPE     = 7'b0110011;
    parameter OPCODE_I_TYPE     = 7'b0010011;
    parameter OPCODE_LOAD       = 7'b0000011;
    parameter OPCODE_STORE      = 7'b0100011;
    parameter OPCODE_BRANCH     = 7'b1100011;
    parameter OPCODE_JAL        = 7'b1101111;
    parameter OPCODE_JALR       = 7'b1100111;
    parameter OPCODE_LUI        = 7'b0110111;
    parameter OPCODE_AUIPC      = 7'b0010111;
    parameter OPCODE_FENCE      = 7'b0001111;
    parameter OPCODE_SYSTEM     = 7'b1110011;

    always @(*) begin
        // Default values
        alu_control = 4'h0;
        rd = instruction[11:7];
        rs1 = instruction[19:15];
        rs2 = instruction[24:20];
        immediate = 32'h0;
        reg_write_enable = 1'b0;
        alu_src = 1'b0;
        branch = 1'b0;
        mem_read = 1'b0;
        mem_write = 1'b0;
        mem_to_reg = 1'b0;
        is_fence = 1'b0;
        is_ecall = 1'b0;
        is_ebreak = 1'b0;
        is_csr = 1'b0;
        csr_op = 3'b000;  // Default CSR operation
         if (is_csr) begin
          
            csr_op = 3'b001; 
        end

        case (opcode)
            OPCODE_R_TYPE: begin
                reg_write_enable = 1'b1;
                alu_src = 1'b0;
                case (funct3)
                    3'b000: alu_control = (funct7 == 7'b0100000) ? 4'h1 : 4'h0;  // SUB : ADD
                    3'b001: alu_control = 4'h5;  // SLL
                    3'b010: alu_control = 4'h8;  // SLT
                    3'b011: alu_control = 4'h9;  // SLTU
                    3'b100: alu_control = 4'h4;  // XOR
                    3'b101: alu_control = (funct7 == 7'b0100000) ? 4'h7 : 4'h6;  // SRA : SRL
                    3'b110: alu_control = 4'h3;  // OR
                    3'b111: alu_control = 4'h2;  // AND
                    default: alu_control = 4'h0;
                endcase
            end

            OPCODE_I_TYPE: begin
                reg_write_enable = 1'b1;
                alu_src = 1'b1;
                immediate = {{20{instruction[31]}}, instruction[31:20]};  // Sign extension
                case (funct3)
                    3'b000: alu_control = 4'h0;  // ADDI
                    3'b010: alu_control = 4'h8;  // SLTI
                    3'b011: alu_control = 4'h9;  // SLTIU
                    3'b100: alu_control = 4'h4;  // XORI
                    3'b110: alu_control = 4'h3;  // ORI
                    3'b111: alu_control = 4'h2;  // ANDI
                    3'b001: alu_control = 4'h5;  // SLLI
                    3'b101: alu_control = (instruction[30]) ? 4'h7 : 4'h6;  // SRAI : SRLI
                    default: alu_control = 4'h0;
                endcase
            end

            OPCODE_LOAD: begin
                reg_write_enable = 1'b1;
                alu_src = 1'b1;
                mem_read = 1'b1;
                mem_to_reg = 1'b1;
                immediate = {{20{instruction[31]}}, instruction[31:20]};
                alu_control = 4'h0;  // ADD for address calculation
            end

            OPCODE_STORE: begin
                alu_src = 1'b1;
                mem_write = 1'b1;
                immediate = {{20{instruction[31]}}, instruction[31:25], instruction[11:7]};
                alu_control = 4'h0;  // ADD for address calculation
            end

            OPCODE_BRANCH: begin
                branch = 1'b1;
                alu_src = 1'b0;
                immediate = {{19{instruction[31]}}, instruction[31], instruction[7], instruction[30:25], instruction[11:8], 1'b0};
                case (funct3)
                    3'b000: alu_control = 4'h1;  // BEQ
                    3'b001: alu_control = 4'h1;  // BNE
                    3'b100: alu_control = 4'h8;  // BLT
                    3'b101: alu_control = 4'h8;  // BGE
                    3'b110: alu_control = 4'h9;  // BLTU
                    3'b111: alu_control = 4'h9;  // BGEU
                    default: alu_control = 4'h0;
                endcase
            end

            OPCODE_JAL: begin
                reg_write_enable = 1'b1;
                alu_src = 1'b0;
                immediate = {{12{instruction[31]}}, instruction[19:12], instruction[20], instruction[30:21], 1'b0};
                alu_control = 4'h0;  // ADD for PC calculation
            end

            OPCODE_JALR: begin
                reg_write_enable = 1'b1;
                alu_src = 1'b1;
                immediate = {{20{instruction[31]}}, instruction[31:20]};
                alu_control = 4'h0;  // ADD for PC calculation
            end

            OPCODE_LUI: begin
                reg_write_enable = 1'b1;
                alu_src = 1'b1;
                immediate = {instruction[31:12], 12'b0};
                alu_control = 4'h0;  // Pass through immediate
            end

            OPCODE_AUIPC: begin
                reg_write_enable = 1'b1;
                alu_src = 1'b1;
                immediate = {instruction[31:12], 12'b0};
                alu_control = 4'h0;  // ADD PC and immediate
            end

            OPCODE_FENCE: begin
                is_fence = 1'b1;
                // FENCE doesn't typically need ALU operations
            end

            OPCODE_SYSTEM: begin
                case (funct3)
                    3'b000: begin
                        case (instruction[31:20])
                            12'b000000000000: is_ecall = 1'b1;   // ECALL
                            12'b000000000001: is_ebreak = 1'b1;  // EBREAK
                            default: ; // Reserved or other system instructions can be handled here
                        endcase
                    end
                    // Optional: Handle CSR instructions if needed
                    default: ;
                endcase
            end

            default: begin
                // Invalid opcode, NOP
                alu_control = 4'h0;
                reg_write_enable = 1'b0;
            end
        endcase
    end
endmodule
// Pipeline_CPU_Core Module
`timescale 1ns / 1ps
module Pipeline_CPU_Core (
    input wire clk,
    input wire rst,
    input wire start_thread,
    input wire [31:0] instruction_input,
    input wire [7:0] instruction_addr,
    input wire instruction_write_enable,
    
    // Execution result output ports
    output wire [31:0] output_mem_addr,
    output wire [31:0] output_mem_data,
    output wire output_mem_write,
    output wire [32*32-1:0] reg_file_out, // Flattened register file (1024 bits)
    
    // CPU state outputs
    output wire [31:0] pc_output,
    output wire [31:0] current_instruction_output,

    // Active thread output
    output wire active_thread
);
  
    parameter N_THREADS = 2;

    // Active thread index (0 or 1)
    reg active_thread_reg;

    // Assign to output
    assign active_thread = active_thread_reg;

    // Program counters for each thread
    reg [31:0] pc [0:N_THREADS-1];
    wire [31:0] next_pc [0:N_THREADS-1];

    // Shared instruction memory
    reg [31:0] instruction_memory [0:255]; // 256 entries
    wire [31:0] instruction [0:N_THREADS-1];

    // Flattened Register File Declaration
    reg [31:0] register_file_flat [0:N_THREADS*32-1]; // Total 64 registers for 2 threads

    // Access Macro for Clarity
    `define REG_FILE(thread, reg) register_file_flat[(thread)*32 + (reg)]

    // Pipeline registers (per thread)
    reg [31:0] IF_ID_pc [0:N_THREADS-1], IF_ID_instr [0:N_THREADS-1], IF_ID_next_pc [0:N_THREADS-1];
    reg [31:0] ID_EX_pc [0:N_THREADS-1], ID_EX_next_pc [0:N_THREADS-1], ID_EX_reg_data1 [0:N_THREADS-1], ID_EX_reg_data2 [0:N_THREADS-1], ID_EX_immediate [0:N_THREADS-1];
    reg [4:0]  ID_EX_rd [0:N_THREADS-1], ID_EX_rs1 [0:N_THREADS-1], ID_EX_rs2 [0:N_THREADS-1];
    reg [3:0]  ID_EX_alu_control [0:N_THREADS-1];
    reg        ID_EX_reg_write_enable [0:N_THREADS-1], ID_EX_alu_src [0:N_THREADS-1], ID_EX_branch [0:N_THREADS-1], ID_EX_mem_read [0:N_THREADS-1], ID_EX_mem_write [0:N_THREADS-1], ID_EX_mem_to_reg [0:N_THREADS-1];
    reg        ID_EX_is_fence [0:N_THREADS-1], ID_EX_is_ecall [0:N_THREADS-1], ID_EX_is_ebreak [0:N_THREADS-1], ID_EX_is_csr [0:N_THREADS-1];
    reg [2:0]  ID_EX_csr_op [0:N_THREADS-1];

    reg [31:0] EX_MEM_alu_result [0:N_THREADS-1], EX_MEM_reg_data2 [0:N_THREADS-1];
    reg [4:0]  EX_MEM_rd [0:N_THREADS-1];
    reg        EX_MEM_reg_write_enable [0:N_THREADS-1], EX_MEM_mem_read [0:N_THREADS-1], EX_MEM_mem_write [0:N_THREADS-1], EX_MEM_mem_to_reg [0:N_THREADS-1];
    reg        EX_MEM_branch_taken [0:N_THREADS-1];
    reg [31:0] EX_MEM_branch_target [0:N_THREADS-1];

    reg [31:0] MEM_WB_mem_data [0:N_THREADS-1], MEM_WB_alu_result [0:N_THREADS-1];
    reg [4:0]  MEM_WB_rd [0:N_THREADS-1];
    reg        MEM_WB_reg_write_enable [0:N_THREADS-1], MEM_WB_mem_to_reg [0:N_THREADS-1];

    // Control signals (per thread)
    wire [3:0]  alu_control [0:N_THREADS-1];
    wire [4:0]  rd [0:N_THREADS-1], rs1 [0:N_THREADS-1], rs2 [0:N_THREADS-1];
    wire [31:0] immediate [0:N_THREADS-1];
    wire        reg_write_enable [0:N_THREADS-1], alu_src [0:N_THREADS-1], branch [0:N_THREADS-1], mem_read [0:N_THREADS-1], mem_write [0:N_THREADS-1], mem_to_reg [0:N_THREADS-1];
    wire        is_fence [0:N_THREADS-1], is_ecall [0:N_THREADS-1], is_ebreak [0:N_THREADS-1], is_csr [0:N_THREADS-1];
    wire [2:0]  csr_op [0:N_THREADS-1];

    // Hazard detection and forwarding units (per thread)
    wire stall [0:N_THREADS-1], IF_ID_write [0:N_THREADS-1], PC_write [0:N_THREADS-1];
    wire [1:0] forwardA [0:N_THREADS-1], forwardB [0:N_THREADS-1];

    // ALU signals (per thread)
    reg [31:0] alu_operand1 [0:N_THREADS-1], alu_operand2 [0:N_THREADS-1];
    wire [31:0] alu_result [0:N_THREADS-1];
    wire        zero_flag [0:N_THREADS-1];

    // Cache and memory signals (shared)
    wire cache_ready;
    wire [31:0] cache_read_data;
    wire cache_hit;

    // Write-back data (per thread)
    reg [31:0] write_back_data [0:N_THREADS-1];

    // EX Stage Control Signals (per thread)
    reg EX_busy [0:N_THREADS-1];
    wire EX_ready [0:N_THREADS-1];

    // Declare L2 and SDRAM Interface Signals
    wire [31:0] l2_addr;
    wire [31:0] l2_write_data;
    wire        l2_mem_read;
    wire        l2_mem_write;
    wire [31:0] l2_read_data;
    wire        l2_ready;
    wire        l2_hit;

    wire [31:0] sdram_addr;
    wire [31:0] sdram_write_data;
    wire        sdram_mem_read;
    wire        sdram_mem_write;
    wire [31:0] sdram_read_data;
    wire        sdram_ready;
    wire        sdram_hit;

    // Declare reg_file_out_flat as an internal wire (flattened register file)
    wire [32*32-1:0] reg_file_out_flat; // 1024 bits

    // Flatten the active thread's register_file into reg_file_out_flat
    genvar reg_idx;
    generate
      for (reg_idx = 0; reg_idx < 32; reg_idx = reg_idx + 1) begin : flatten_registers
            assign reg_file_out_flat[reg_idx*32 +: 32] = `REG_FILE(active_thread_reg, reg_idx);
      end
    endgenerate

    integer i, j;

    // Active thread selection logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            active_thread_reg <= 0;
        end else begin
            active_thread_reg <= start_thread;
        end
    end

    reg mem_initialized;

    // Instruction Memory Initialization and Write Logic
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            mem_initialized <= 0;
            for (integer i = 0; i < 256; i = i + 1) begin
                instruction_memory[i] <= 32'h00000013; // NOP instruction (addi x0, x0, 0)
            end
        end else if (!mem_initialized) begin
            if (instruction_write_enable) begin
                instruction_memory[instruction_addr] <= instruction_input;
                if (instruction_addr == 8'hFF) begin // Last address
                    mem_initialized <= 1;
                end
            end
        end else if (instruction_write_enable) begin
            instruction_memory[instruction_addr] <= instruction_input;
        end
    end

    // Program counter update (per thread)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < N_THREADS; i = i + 1) begin
                pc[i] <= 32'h0;
            end
        end else if (PC_write[active_thread_reg]) begin
            pc[active_thread_reg] <= next_pc[active_thread_reg];
        end
    end

    // Instruction fetch (per thread)
    genvar idx;
    generate
        for (idx = 0; idx < N_THREADS; idx = idx + 1) begin : instruction_fetch
            assign instruction[idx] = instruction_memory[pc[idx][9:2]];
        end
    endgenerate

    // IF/ID pipeline register update (per thread)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < N_THREADS; i = i + 1) begin
                IF_ID_pc[i] <= 32'h0;
                IF_ID_instr[i] <= 32'h00000013; // NOP
                IF_ID_next_pc[i] <= 32'h0;
            end
        end else if (IF_ID_write[active_thread_reg]) begin
            IF_ID_pc[active_thread_reg] <= pc[active_thread_reg];
            IF_ID_instr[active_thread_reg] <= instruction[active_thread_reg];
            IF_ID_next_pc[active_thread_reg] <= pc[active_thread_reg] + 4;
        end
    end

    // Instruction Decode (per thread)
    generate
         genvar idx_inner; // New genvar declaration
         for (idx_inner = 0; idx_inner < N_THREADS; idx_inner = idx_inner + 1) begin : thread_loop
            Instruction_Decoder decoder (
                .instruction(IF_ID_instr[idx_inner]),
                .alu_control(alu_control[idx_inner]),
                .rd(rd[idx_inner]),
                .rs1(rs1[idx_inner]),
                .rs2(rs2[idx_inner]),
                .immediate(immediate[idx_inner]),
                .reg_write_enable(reg_write_enable[idx_inner]),
                .alu_src(alu_src[idx_inner]),
                .branch(branch[idx_inner]),
                .mem_read(mem_read[idx_inner]),
                .mem_write(mem_write[idx_inner]),
                .mem_to_reg(mem_to_reg[idx_inner]),
                .is_fence(is_fence[idx_inner]),
                .is_ecall(is_ecall[idx_inner]),
                .is_ebreak(is_ebreak[idx_inner]),
                .is_csr(is_csr[idx_inner]),
                .csr_op(csr_op[idx_inner])
            );

            // Hazard Detection Unit
            Hazard_Detection_Unit hazard_unit (
                .ID_EX_rd(ID_EX_rd[idx_inner]),
                .IF_ID_rs1(IF_ID_instr[idx_inner][19:15]),
                .IF_ID_rs2(IF_ID_instr[idx_inner][24:20]),
                .ID_EX_mem_read(ID_EX_mem_read[idx_inner]),
                .stall(stall[idx_inner]),
                .IF_ID_write(IF_ID_write[idx_inner]),
                .PC_write(PC_write[idx_inner])
            );

            // Forwarding Unit
            Forwarding_Unit forwarding_unit (
                .ID_EX_rs1(ID_EX_rs1[idx_inner]),
                .ID_EX_rs2(ID_EX_rs2[idx_inner]),
                .EX_MEM_rd(EX_MEM_rd[idx_inner]),
                .MEM_WB_rd(MEM_WB_rd[idx_inner]),
                .EX_MEM_reg_write(EX_MEM_reg_write_enable[idx_inner]),
                .MEM_WB_reg_write(MEM_WB_reg_write_enable[idx_inner]),
                .forwardA(forwardA[idx_inner]),
                .forwardB(forwardB[idx_inner])
            );

            // ALU Instance
            ALU alu (
                .operand1(alu_operand1[idx_inner]),
                .operand2(alu_operand2[idx_inner]),
                .alu_control(ID_EX_alu_control[idx_inner]),
                .alu_result(alu_result[idx_inner]),
                .zero_flag(zero_flag[idx_inner])
            );
        end
    endgenerate

    // ID/EX pipeline register update (per thread)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < N_THREADS; i = i + 1) begin
                ID_EX_pc[i] <= 32'h0;
                ID_EX_next_pc[i] <= 32'h0;
                ID_EX_reg_data1[i] <= 32'h0;
                ID_EX_reg_data2[i] <= 32'h0;
                ID_EX_immediate[i] <= 32'h0;
                ID_EX_rd[i] <= 5'h0;
                ID_EX_rs1[i] <= 5'h0;
                ID_EX_rs2[i] <= 5'h0;
                ID_EX_alu_control[i] <= 4'h0;
                ID_EX_reg_write_enable[i] <= 1'h0;
                ID_EX_alu_src[i] <= 1'h0;
                ID_EX_mem_read[i] <= 1'h0;
                ID_EX_mem_write[i] <= 1'h0;
                ID_EX_mem_to_reg[i] <= 1'h0;
                ID_EX_branch[i] <= 1'h0;
                ID_EX_is_fence[i] <= 1'b0;
                ID_EX_is_ecall[i] <= 1'b0;
                ID_EX_is_ebreak[i] <= 1'b0;
                ID_EX_is_csr[i] <= 1'b0;
                ID_EX_csr_op[i] <= 3'b000;
            end
        end else if (!stall[active_thread_reg]) begin
            ID_EX_pc[active_thread_reg] <= IF_ID_pc[active_thread_reg];
            ID_EX_next_pc[active_thread_reg] <= IF_ID_next_pc[active_thread_reg];
            ID_EX_reg_data1[active_thread_reg] <= `REG_FILE(active_thread_reg, rs1[active_thread_reg]);
            ID_EX_reg_data2[active_thread_reg] <= `REG_FILE(active_thread_reg, rs2[active_thread_reg]);
            ID_EX_immediate[active_thread_reg] <= immediate[active_thread_reg];
            ID_EX_rd[active_thread_reg] <= rd[active_thread_reg];
            ID_EX_rs1[active_thread_reg] <= rs1[active_thread_reg];
            ID_EX_rs2[active_thread_reg] <= rs2[active_thread_reg];
            ID_EX_alu_control[active_thread_reg] <= alu_control[active_thread_reg];
            ID_EX_reg_write_enable[active_thread_reg] <= reg_write_enable[active_thread_reg];
            ID_EX_alu_src[active_thread_reg] <= alu_src[active_thread_reg];
            ID_EX_mem_read[active_thread_reg] <= mem_read[active_thread_reg];
            ID_EX_mem_write[active_thread_reg] <= mem_write[active_thread_reg];
            ID_EX_mem_to_reg[active_thread_reg] <= mem_to_reg[active_thread_reg];
            ID_EX_branch[active_thread_reg] <= branch[active_thread_reg];
            ID_EX_is_fence[active_thread_reg] <= is_fence[active_thread_reg];
            ID_EX_is_ecall[active_thread_reg] <= is_ecall[active_thread_reg];
            ID_EX_is_ebreak[active_thread_reg] <= is_ebreak[active_thread_reg];
            ID_EX_is_csr[active_thread_reg] <= is_csr[active_thread_reg];
            ID_EX_csr_op[active_thread_reg] <= csr_op[active_thread_reg];
        end else begin
            // Insert NOP on stall
            ID_EX_reg_write_enable[active_thread_reg] <= 1'b0;
            ID_EX_mem_read[active_thread_reg] <= 1'b0;
            ID_EX_mem_write[active_thread_reg] <= 1'b0;
            ID_EX_branch[active_thread_reg] <= 1'b0;
            ID_EX_mem_to_reg[active_thread_reg] <= 1'b0;
            ID_EX_alu_src[active_thread_reg] <= 1'b0;
            ID_EX_alu_control[active_thread_reg] <= 4'b0000;
            ID_EX_is_fence[active_thread_reg] <= 1'b0;
            ID_EX_is_ecall[active_thread_reg] <= 1'b0;
            ID_EX_is_ebreak[active_thread_reg] <= 1'b0;
            ID_EX_is_csr[active_thread_reg] <= 1'b0;
            ID_EX_csr_op[active_thread_reg] <= 3'b000;
        end
    end

    // Execution stage (per thread)
    always @(*) begin
        for (i = 0; i < N_THREADS; i = i + 1) begin
            // Operand 1 forwarding
            case (forwardA[i])
                2'b00: alu_operand1[i] = ID_EX_reg_data1[i];
                2'b10: alu_operand1[i] = EX_MEM_alu_result[i];
                2'b01: alu_operand1[i] = write_back_data[i];
                default: alu_operand1[i] = ID_EX_reg_data1[i];
            endcase

            // Operand 2 forwarding
            if (ID_EX_alu_src[i]) begin
                alu_operand2[i] = ID_EX_immediate[i];
            end else begin
                case (forwardB[i])
                    2'b00: alu_operand2[i] = ID_EX_reg_data2[i];
                    2'b10: alu_operand2[i] = EX_MEM_alu_result[i];
                    2'b01: alu_operand2[i] = write_back_data[i];
                    default: alu_operand2[i] = ID_EX_reg_data2[i];
                endcase
            end
        end
    end

    // EX stage control (per thread)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < N_THREADS; i = i + 1) begin
                EX_busy[i] <= 1'b0;
            end
        end else begin
            for (i = 0; i < N_THREADS; i = i + 1) begin
                EX_busy[i] <= 1'b0; // No extension-specific operations
            end
        end
    end

    generate
        for (idx = 0; idx < N_THREADS; idx = idx + 1) begin : ex_ready_gen
            assign EX_ready[idx] = 1'b1; // Always ready as no extension operations
        end
    endgenerate

    // EX/MEM pipeline register update (per thread)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < N_THREADS; i = i + 1) begin
                EX_MEM_alu_result[i] <= 32'h0;
                EX_MEM_reg_data2[i] <= 32'h0;
                EX_MEM_rd[i] <= 5'h0;
                EX_MEM_reg_write_enable[i] <= 1'h0;
                EX_MEM_mem_read[i] <= 1'h0;
                EX_MEM_mem_write[i] <= 1'h0;
                EX_MEM_mem_to_reg[i] <= 1'h0;
                EX_MEM_branch_taken[i] <= 1'h0;
                EX_MEM_branch_target[i] <= 32'h0;
            end
        end else if (EX_ready[active_thread_reg]) begin
            EX_MEM_alu_result[active_thread_reg] <= alu_result[active_thread_reg];
            EX_MEM_reg_data2[active_thread_reg] <= ID_EX_reg_data2[active_thread_reg];
            EX_MEM_rd[active_thread_reg] <= ID_EX_rd[active_thread_reg];
            EX_MEM_reg_write_enable[active_thread_reg] <= ID_EX_reg_write_enable[active_thread_reg];
            EX_MEM_mem_read[active_thread_reg] <= ID_EX_mem_read[active_thread_reg];
            EX_MEM_mem_write[active_thread_reg] <= ID_EX_mem_write[active_thread_reg];
            EX_MEM_mem_to_reg[active_thread_reg] <= ID_EX_mem_to_reg[active_thread_reg];
            EX_MEM_branch_taken[active_thread_reg] <= ID_EX_branch[active_thread_reg] & zero_flag[active_thread_reg];
            EX_MEM_branch_target[active_thread_reg] <= ID_EX_pc[active_thread_reg] + (ID_EX_immediate[active_thread_reg] << 1);
        end
    end

    // Memory stage (shared cache and hierarchy)
    // Dynamic Output Allocation module instance
    Dynamic_Output_Allocation output_allocator (
        .clk(clk),
        .rst(rst),
        .instruction(IF_ID_instr[active_thread_reg]),
        .register_file(reg_file_out_flat),
        .mem_addr(output_mem_addr),
        .mem_data(output_mem_data),
        .mem_write(output_mem_write)
    );

    // L1 Cache instance
    L1_Cache l1 (
        .clk(clk),
        .rst(rst),
        .addr(EX_MEM_alu_result[active_thread_reg]),
        .write_data(EX_MEM_reg_data2[active_thread_reg]),
        .mem_read(EX_MEM_mem_read[active_thread_reg]),
        .mem_write(EX_MEM_mem_write[active_thread_reg]),
        .read_data(cache_read_data),
        .hit(cache_hit),
        .ready(cache_ready),
        // L2 Cache Interface
        .l2_addr(l2_addr),
        .l2_write_data(l2_write_data),
        .l2_mem_read(l2_mem_read),
        .l2_mem_write(l2_mem_write),
        .l2_read_data(l2_read_data),
        .l2_ready(l2_ready),
        .l2_hit(l2_hit)
    );

    // L2_Cache Instantiation
    L2_Cache l2 (
        .clk(clk),
        .rst(rst),
        .addr(l2_addr),                   // 32-bit
        .write_data(l2_write_data),       // 32-bit
        .mem_read(l2_mem_read),           // 1-bit
        .mem_write(l2_mem_write),         // 1-bit
        .read_data(l2_read_data),         // 32-bit
        .hit(l2_hit),                     // 1-bit
        .ready(l2_ready),                 // 1-bit
        // SDRAM Interface
        .sdram_addr(sdram_addr),          // 32-bit
        .sdram_write_data(sdram_write_data), // 32-bit
        .sdram_mem_read(sdram_mem_read),      // 1-bit
        .sdram_mem_write(sdram_mem_write),    // 1-bit
        .sdram_read_data(sdram_read_data),    // 32-bit
        .sdram_ready(sdram_ready),            // 1-bit
        .sdram_hit(sdram_hit)                 // 1-bit (always 1)
    );

    // SDRAM_Controller Instantiation
    SDRAM_Controller sdram (
        .clk(clk),
        .rst(rst),
        .addr(sdram_addr),               // 32-bit
        .write_data(sdram_write_data),   // 32-bit
        .mem_read(sdram_mem_read),       // 1-bit
        .mem_write(sdram_mem_write),     // 1-bit
        .read_data(sdram_read_data),     // 32-bit
        .ready(sdram_ready),             // 1-bit
        .hit(sdram_hit)                  // 1-bit (always 1)
    );

    // MEM/WB pipeline register update (per thread)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < N_THREADS; i = i + 1) begin
                MEM_WB_mem_data[i] <= 32'h0;
                MEM_WB_alu_result[i] <= 32'h0;
                MEM_WB_rd[i] <= 5'h0;
                MEM_WB_reg_write_enable[i] <= 1'h0;
                MEM_WB_mem_to_reg[i] <= 1'h0;
            end
        end else begin
            MEM_WB_mem_data[active_thread_reg] <= cache_read_data;
            MEM_WB_alu_result[active_thread_reg] <= EX_MEM_alu_result[active_thread_reg];
            MEM_WB_rd[active_thread_reg] <= EX_MEM_rd[active_thread_reg];
            MEM_WB_reg_write_enable[active_thread_reg] <= EX_MEM_reg_write_enable[active_thread_reg];
            MEM_WB_mem_to_reg[active_thread_reg] <= EX_MEM_mem_to_reg[active_thread_reg];
        end
    end

    // Write-back data logic (per thread)
    always @(*) begin
        for (i = 0; i < N_THREADS; i = i + 1) begin
            write_back_data[i] = MEM_WB_mem_to_reg[i] ? MEM_WB_mem_data[i] : MEM_WB_alu_result[i];
        end
    end

    // Register file update (per thread)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < N_THREADS; i = i + 1) begin
                for (j = 0; j < 32; j = j + 1) begin
                    if (j == 2) begin
                        `REG_FILE(i, j) <= 32'h80000000; // Stack Pointer (x2) 초기화
                    end else if (j == 3) begin
                        `REG_FILE(i, j) <= 32'h80001000; // Global Pointer (x3) 초기화
                    end else if (j == 0) begin
                        `REG_FILE(i, j) <= 32'h00000000; // x0 항상 0
                    end else begin
                        `REG_FILE(i, j) <= 32'h0; 
                    end
                end
            end
            // Diagnostic Display After Initialization
            for (i = 0; i < N_THREADS; i = i + 1) begin
                for (j = 0; j < 32; j = j + 1) begin
                    
                end
            end
        end else begin
            if (MEM_WB_reg_write_enable[active_thread_reg] && MEM_WB_rd[active_thread_reg] != 5'h0) begin
                `REG_FILE(active_thread_reg, MEM_WB_rd[active_thread_reg]) <= write_back_data[active_thread_reg];
                
            end
        end
    end

    // Next PC calculation (per thread)
    generate
        for (idx = 0; idx < N_THREADS; idx = idx + 1) begin : next_pc_gen
            assign next_pc[idx] = EX_MEM_branch_taken[idx] ? EX_MEM_branch_target[idx] : pc[idx] + 4;
        end
    endgenerate

    // Output assignments
    assign pc_output = pc[active_thread_reg];
    assign current_instruction_output = instruction[active_thread_reg];

endmodule
// Top-level module
`timescale 1ns / 1ps

module Hyperthreaded_CPU (
    input wire clk,
    input wire rst,
    
    // Instruction loading inputs
    input wire [31:0] instruction_input,
    input wire [7:0] instruction_addr,
    input wire instruction_write_enable,
    
    // Execution result outputs
    output wire [31:0] output_mem_addr,
    output wire [31:0] output_mem_data,
    output wire output_mem_write,
    
    // CPU state outputs
    output wire [31:0] pc_output,
    output wire [31:0] current_instruction_output,

    // Active thread output
    output wire active_thread,

    // Optional: Debug mode input
    input wire debug_mode
);
    // Declare reg_file_out_flat as an internal wire array
    wire [32*32-1:0] reg_file_out_flat; // 1024-bit wire

    // Thread control signal
    reg start_thread;

    // Instantiate Pipeline_CPU_Core
    Pipeline_CPU_Core cpu_core (
        .clk(clk),
        .rst(rst),
        .start_thread(start_thread),
        .instruction_input(instruction_input),
        .instruction_addr(instruction_addr),
        .instruction_write_enable(instruction_write_enable),
        .output_mem_addr(output_mem_addr),
        .output_mem_data(output_mem_data),
        .output_mem_write(output_mem_write),
        .reg_file_out(reg_file_out_flat), 
        .pc_output(pc_output),
        .current_instruction_output(current_instruction_output),
        .active_thread(active_thread)
        
    );

    // Thread switching logic
    integer cycle_count;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            start_thread <= 1'b0;
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (cycle_count == 10) begin // Switch thread every 10 cycles
                start_thread <= ~start_thread;
                cycle_count <= 0;
            end
        end
    end


    integer k;
    always @(posedge clk) begin
        if (debug_mode) begin
            for (k = 0; k < 32; k = k + 1) begin

            end
        end
    end

endmodule


module L1_Cache (
    input clk,
    input rst,
    input [31:0] addr,
    input [31:0] write_data,
    input mem_read,
    input mem_write,
    output reg [31:0] read_data,
    output reg hit,
    output reg ready,
    
    output reg [31:0] l2_addr,
    output reg [31:0] l2_write_data,
    output reg l2_mem_read,
    output reg l2_mem_write,
    input [31:0] l2_read_data,
    input l2_ready,
    input l2_hit
);
    
    reg [31:0] cache_data [0:63];
    reg [19:0] cache_tag [0:63];
    reg cache_valid [0:63];
    wire [5:0] index;
    wire [19:0] tag;
    integer i;

    assign index = addr[7:2];
    assign tag = addr[31:12];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            // 캐시 초기화
            for (i = 0; i < 64; i = i + 1) begin
                cache_data[i] <= 32'h0;
                cache_tag[i] <= 20'h0;
                cache_valid[i] <= 1'b0;
            end
            ready <= 1'b1;
            hit <= 1'b0;
            read_data <= 32'h0;
        end else begin
            if (mem_read || mem_write) begin
                if (cache_valid[index] && cache_tag[index] == tag) begin
                    // 캐시 히트
                    hit <= 1'b1;
                    ready <= 1'b1;
                    if (mem_read) begin
                        read_data <= cache_data[index];
                    end
                    if (mem_write) begin
                        cache_data[index] <= write_data;
                    end
                end else begin
                    // 캐시 미스 -> L2 캐시에 요청
                    hit <= 1'b0;
                    ready <= 1'b0;
                    l2_addr <= addr;
                    l2_write_data <= write_data;
                    l2_mem_read <= mem_read;
                    l2_mem_write <= mem_write;
                    if (l2_ready) begin
                        if (l2_mem_read) begin
                            cache_data[index] <= l2_read_data;
                            cache_tag[index] <= tag;
                            cache_valid[index] <= 1'b1;
                            read_data <= l2_read_data;
                        end
                        ready <= 1'b1;
                    end
                end
            end else begin
                ready <= 1'b1;
                hit <= 1'b0;
            end
        end
    end
endmodule
module Dynamic_Output_Allocation (
    input wire clk,
    input wire rst,
    input wire [31:0] instruction,
  input wire [32*32-1:0] register_file,  // Flattened single thread's register file (1024 bits)
    output reg [31:0] mem_addr,
    output reg [31:0] mem_data,
    output reg mem_write
);
    
    // Unpack the flattened register_file into individual registers
    reg [31:0] reg_file [0:31];
    integer i;

    always @(*) begin
        for (i = 0; i < 32; i = i + 1) begin
            reg_file[i] = register_file[i*32 +: 32];
        end
    end

    // Instruction parsing
    wire [6:0] opcode;
    wire [4:0] rd, rs1, rs2;
    wire [11:0] imm;

    assign opcode = instruction[6:0];
    assign rd = instruction[11:7];
    assign rs1 = instruction[19:15];
    assign rs2 = instruction[24:20];
    assign imm = instruction[31:20];

    // State machine
    reg [1:0] state;
    parameter IDLE = 2'b00, PARSE = 2'b01, WRITE = 2'b10;

    // Temporary storage
    reg [31:0] temp_addr;
    reg [31:0] temp_data;
    reg [4:0] temp_reg;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            mem_addr <= 32'b0;
            mem_data <= 32'b0;
            mem_write <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    if (opcode == 7'b0010011 && rs1 == 5'b0) begin  // ADDI with x0 as source
                        state <= PARSE;
                    end
                end

                PARSE: begin
                    temp_reg <= rd;
                    temp_addr <= {20'b0, imm};  // Use immediate as memory address
                    state <= WRITE;
                end

                WRITE: begin
                    mem_addr <= temp_addr;
                    mem_data <= reg_file[temp_reg];
                    mem_write <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule

module L2_Cache (
    input clk,
    input rst,
    input [31:0] addr,
    input [31:0] write_data,
    input mem_read,
    input mem_write,
    output reg [31:0] read_data,
    output reg hit,
    output reg ready,
    // SDRAM 인터페이스
    output reg [31:0] sdram_addr,
    output reg [31:0] sdram_write_data,
    output reg sdram_mem_read,
    output reg sdram_mem_write,
    input [31:0] sdram_read_data,
    input sdram_ready,
    input sdram_hit
);
    // L2 캐시 구성: 256개의 라인
    reg [31:0] cache_data [0:255];
    reg [19:0] cache_tag [0:255];
    reg cache_valid [0:255];
    wire [7:0] index;
    wire [19:0] tag;
    integer i;

    assign index = addr[9:2];
    assign tag = addr[31:12];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            // 캐시 초기화
            for (i = 0; i < 256; i = i + 1) begin
                cache_data[i] <= 32'h0;
                cache_tag[i] <= 20'h0;
                cache_valid[i] <= 1'b0;
            end
            ready <= 1'b1;
            hit <= 1'b0;
            read_data <= 32'h0;
        end else begin
            if (mem_read || mem_write) begin
                if (cache_valid[index] && cache_tag[index] == tag) begin
                    // 캐시 히트
                    hit <= 1'b1;
                    ready <= 1'b1;
                    if (mem_read) begin
                        read_data <= cache_data[index];
                    end
                    if (mem_write) begin
                        cache_data[index] <= write_data;
                    end
                end else begin
                    // 캐시 미스 -> SDRAM에 요청
                    hit <= 1'b0;
                    ready <= 1'b0;
                    sdram_addr <= addr;
                    sdram_write_data <= write_data;
                    sdram_mem_read <= mem_read;
                    sdram_mem_write <= mem_write;
                    if (sdram_ready) begin
                        if (sdram_mem_read) begin
                            cache_data[index] <= sdram_read_data;
                            cache_tag[index] <= tag;
                            cache_valid[index] <= 1'b1;
                            read_data <= sdram_read_data;
                        end
                        ready <= 1'b1;
                    end
                end
            end else begin
                ready <= 1'b1;
                hit <= 1'b0;
            end
        end
    end
endmodule
module SDRAM_Controller (
    input clk,
    input rst,
    input [31:0] addr,
    input [31:0] write_data,
    input mem_read,
    input mem_write,
    output reg [31:0] read_data,
    output reg ready,
    output reg hit  // SDRAM에서는 항상 히트로 처리
);
   
    reg [31:0] sdram_mem [0:1023];
    integer i;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ready <= 1'b1;
            hit <= 1'b1;
            read_data <= 32'h0;
            // 메모리 초기화
            for (i = 0; i < 1024; i = i + 1) begin
                sdram_mem[i] <= 32'h0;
            end
        end else begin
            ready <= 1'b1; 
            if (mem_read) begin
                read_data <= sdram_mem[addr[11:2]];
            end else if (mem_write) begin
                sdram_mem[addr[11:2]] <= write_data;
            end
        end
    end
endmodule
module Branch_Predictor (
    input clk,
    input rst,
    input branch_instr,
    input actual_taken,
    output reg predicted_taken
);
    reg predictor; // 1비트 예측기: 1이면 taken, 0이면 not taken

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            predictor <= 1'b0;
        end else if (branch_instr) begin
            predictor <= actual_taken;
        end
    end

    always @(*) begin
        predicted_taken = predictor;
    end
endmodule
module Cache_Line (
    input clk,
    input rst,
    input valid_in,
    input [19:0] tag_in,
    input [31:0] data_in,
    input write_enable,
    output reg valid,
    output reg [19:0] tag,
    output reg [31:0] data
);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid <= 1'b0;
            tag <= 20'h0;
            data <= 32'h0;
        end else if (write_enable) begin
            valid <= valid_in;
            tag <= tag_in;
            data <= data_in;
        end
    end
endmodule

// Hazard_Detection_Unit Module
module Hazard_Detection_Unit (
    input [4:0] ID_EX_rd,
    input [4:0] IF_ID_rs1,
    input [4:0] IF_ID_rs2,
    input ID_EX_mem_read,
    output reg stall,
    output reg IF_ID_write,
    output reg PC_write
);
    always @(*) begin
        stall = 1'b0;
        IF_ID_write = 1'b1;
        PC_write = 1'b1;

        // Detect load-use hazard
        if (ID_EX_mem_read && 
           ((ID_EX_rd == IF_ID_rs1) || (ID_EX_rd == IF_ID_rs2))) begin
            stall = 1'b1;
            IF_ID_write = 1'b0;
            PC_write = 1'b0;
        end
    end
endmodule

module Forwarding_Unit (
    input [4:0] ID_EX_rs1,
    input [4:0] ID_EX_rs2,
    input [4:0] EX_MEM_rd,
    input [4:0] MEM_WB_rd,
    input EX_MEM_reg_write,
    input MEM_WB_reg_write,
    output reg [1:0] forwardA,
    output reg [1:0] forwardB
);
    always @(*) begin
   
        forwardA = 2'b00;
        forwardB = 2'b00;

        // Forwarding for rs1
        if (EX_MEM_reg_write && (EX_MEM_rd != 5'h0) && (EX_MEM_rd == ID_EX_rs1))
            forwardA = 2'b10;
        else if (MEM_WB_reg_write && (MEM_WB_rd != 5'h0) && (MEM_WB_rd == ID_EX_rs1))
            forwardA = 2'b01;

        // Forwarding for rs2
        if (EX_MEM_reg_write && (EX_MEM_rd != 5'h0) && (EX_MEM_rd == ID_EX_rs2))
            forwardB = 2'b10;
        else if (MEM_WB_reg_write && (MEM_WB_rd != 5'h0) && (MEM_WB_rd == ID_EX_rs2))
            forwardB = 2'b01;
    end
endmodule
